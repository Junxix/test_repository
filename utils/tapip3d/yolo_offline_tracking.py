import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import functools
from PIL import Image
from tqdm import tqdm
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

# YOLO相关导入
from ultralytics import YOLO

sys.path.append('/home/jingjing/workspace/su1/TAPIP3D/')
from utils.inference_utils import load_model, inference_with_mask, get_grid_queries, inference
from utils.common_utils import batch_unproject, batch_project, setup_logger
import models
sys.path.append('/home/jingjing/workspace/sam2')  
from sam2.build_sam import build_sam2_video_predictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class ImprovedSAM2TAPIP3DIntegration:
    def __init__(self, sam2_checkpoint: str, sam2_config: str, tapip3d_checkpoint: str, 
                 yolo_model: str = "yolov8n.pt", device: str = "auto"):
        self.device = self._setup_device(device)
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.tapip3d_checkpoint = tapip3d_checkpoint
        
        self._init_sam2()
        self._init_tapip3d()
        self._init_yolo(yolo_model)
        
        self.inference_state = None
        self.video_dir = None  
        
        self.video_data = None
        self.depth_data = None
        self.intrinsics = None
        self.extrinsics = None
        
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        return device
    
    def _init_sam2(self):
        if self.device.startswith("cuda"):
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        self.sam2_predictor = build_sam2_video_predictor(
            self.sam2_config, 
            self.sam2_checkpoint, 
            device=self.device
        )
        print(f"SAM2模型已加载到 {self.device}")
    
    def _init_tapip3d(self):
        self.tapip3d_model = load_model(self.tapip3d_checkpoint)
        self.tapip3d_model.to(self.device)
        self.tapip3d_model.eval()
        
        if hasattr(self.tapip3d_model, "set_eval_mode"):
            self.tapip3d_model.set_eval_mode("raw")
        
        print(f"TAPIP3D模型已加载到 {self.device}")
    
    def _init_yolo(self, yolo_model: str):
        """初始化YOLO模型"""
        self.yolo_model = YOLO(yolo_model)
        print(f"YOLO模型已加载: {yolo_model}")
    
    def load_data(self, input_path: str, target_resolution: Optional[Tuple[int, int]] = None):
        print(f"从 {input_path} 加载数据...")
        
        if input_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
            self._load_video_data(input_path, target_resolution)
        elif input_path.endswith('.npz'):
            self._load_npz_data(input_path, target_resolution)
        else:
            raise ValueError(f"不支持的输入文件格式: {input_path}")
    
    def _load_video_data(self, video_path: str, target_resolution: Optional[Tuple[int, int]]):
        video = self._read_video(video_path)
        
        if target_resolution:
            video = self._resize_video(video, target_resolution)

        self.video_data = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        self._estimate_depth_and_camera_params(video)
        
        self._move_data_to_device()
    
    def _load_npz_data(self, npz_path: str, target_resolution: Optional[Tuple[int, int]]):
        data = np.load(npz_path)
        
        if 'video' not in data:
            raise ValueError("NPZ文件必须包含'video'键")
        
        video = data['video']
        if video.ndim == 4 and video.shape[-1] == 3:
            self.video_data = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        else:
            raise ValueError(f"视频形状 {video.shape} 不支持")
        
        if 'depths' in data:
            self.depth_data = torch.from_numpy(data['depths']).float()
        
        self.intrinsics = torch.from_numpy(data.get('intrinsics', self._create_default_intrinsics())).float()
        
        T = self.video_data.shape[0]
        self.extrinsics = torch.eye(4).unsqueeze(0).repeat(T, 1, 1).float()
    
        if target_resolution:
            self._resize_data(target_resolution)
        
        self._move_data_to_device()
    
    def _read_video(self, video_path: str) -> np.ndarray:
        import av
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        return np.stack(frames)
    
    def _resize_video(self, video: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
        H, W = target_resolution
        return np.stack([cv2.resize(frame, (W, H)) for frame in video])
    
    def _estimate_depth_and_camera_params(self, video: np.ndarray):
        T, H, W, _ = video.shape
        self.depth_data = torch.ones(T, H, W) * 5.0
        self.intrinsics = torch.from_numpy(self._create_default_intrinsics()).float()
        
        self.extrinsics = torch.eye(4).unsqueeze(0).repeat(T, 1, 1).float()
    
    def _create_default_intrinsics(self) -> np.ndarray:
        T, _, H, W = self.video_data.shape
        f = min(H, W) / (2 * np.tan(np.pi / 6))
        intrinsics = np.eye(3)
        intrinsics[0, 0] = f
        intrinsics[1, 1] = f
        intrinsics[0, 2] = W / 2
        intrinsics[1, 2] = H / 2
        return np.tile(intrinsics[None], (T, 1, 1))
    
    def _resize_data(self, target_resolution: Tuple[int, int]):
        H, W = target_resolution
        
        self.video_data = torch.nn.functional.interpolate(
            self.video_data, size=(H, W), mode='bilinear', align_corners=False
        )
        
        self.depth_data = torch.nn.functional.interpolate(
            self.depth_data.unsqueeze(1), size=(H, W), mode='nearest'
        ).squeeze(1)
        
        orig_H, orig_W = self.depth_data.shape[1:3]
        scale_x = W / orig_W
        scale_y = H / orig_H
        self.intrinsics[:, 0, :] *= scale_x
        self.intrinsics[:, 1, :] *= scale_y
    
    def _move_data_to_device(self):
        self.video_data = self.video_data.to(self.device)
        self.depth_data = self.depth_data.to(self.device)
        self.intrinsics = self.intrinsics.to(self.device)
        self.extrinsics = self.extrinsics.to(self.device)
    
    def prepare_sam2_video(self, temp_dir: Optional[str] = None) -> str:        
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        video_dir = Path(temp_dir)
        video_dir.mkdir(exist_ok=True)
        
        T = self.video_data.shape[0]
        for t in range(T):
            frame = self.video_data[t].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame_path = video_dir / f"{t:06d}.jpg"
            Image.fromarray(frame).save(frame_path)
        
        self.video_dir = str(video_dir)
        
        return str(video_dir)
    
    def _reset_sam2_state_for_new_target(self):
        print("重置SAM2状态以处理新目标")
        
        if self.video_dir is not None:
            self.inference_state = self.sam2_predictor.init_state(video_path=self.video_dir)
            print("SAM2状态已初始化")
        else:
            raise ValueError("video_dir为None，请先运行prepare_sam2_video()")
    
    def detect_objects_yolo(self, frame_idx: int = 0, conf_threshold: float = 0.25, 
                           class_filter: Optional[List[int]] = None) -> List[Dict]:
        """
        使用YOLO检测第一帧中的对象
        
        Args:
            frame_idx: 帧索引
            conf_threshold: 置信度阈值
            class_filter: 要检测的类别ID列表，None表示检测所有类别
        
        Returns:
            检测结果列表，每个元素包含bbox、confidence、class_id、class_name
        """
        frame = self.video_data[frame_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        # YOLO推理
        results = self.yolo_model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                
                # 类别过滤
                if class_filter is not None and class_id not in class_filter:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        print(f"在第{frame_idx}帧检测到 {len(detections)} 个对象")
        return detections
    
    def visualize_detections(self, frame_idx: int, detections: List[Dict], 
                            save_path: Optional[str] = None):
        """可视化YOLO检测结果"""
        frame = self.video_data[frame_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        ax = plt.gca()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 绘制边界框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # 添加标签
            label = f"[{i}] {class_name}: {confidence:.2f}"
            ax.text(x1, y1-5, label, color='red', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"YOLO检测结果 - 第{frame_idx}帧 (共{len(detections)}个对象)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"检测结果已保存到: {save_path}")
        
        # plt.show()
    
    def auto_segment_with_bbox(self, frame_idx: int, bbox: List[float], 
                               target_id: int = 1) -> Optional[torch.Tensor]:
        """
        使用bounding box自动生成mask
        
        Args:
            frame_idx: 帧索引
            bbox: [x1, y1, x2, y2] 格式的边界框
            target_id: 目标ID
        
        Returns:
            生成的mask (torch.Tensor)
        """
        self._reset_sam2_state_for_new_target()
        
        # 将bbox传递给SAM2
        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=target_id,
            box=np.array(bbox)
        )
        
        # 提取mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        if len(mask.shape) == 3:
            mask = mask[0]
        
        print(f"目标 {target_id} 的mask已生成，bbox: {bbox}")
        return torch.from_numpy(mask.astype(bool))
    
    def visualize_mask(self, frame_idx: int, mask: torch.Tensor, bbox: Optional[List[float]] = None,
                      title: str = "Mask可视化", save_path: Optional[str] = None):
        """可视化mask"""
        frame = self.video_data[frame_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(frame)
        
        # 显示mask
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        h, w = mask_np.shape[-2:]
        color = np.array([1, 0, 0, 0.6])
        mask_image = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image, alpha=0.6)
        
        # 如果有bbox，也显示出来
        if bbox is not None:
            ax = plt.gca()
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Mask可视化已保存到: {save_path}")
        
        # plt.show()

    def track_with_tapip3d(self, mask: torch.Tensor, grid_size: int = 0, 
                          num_iters: int = 6, vis_threshold: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(self.device)

        query_points = self._generate_query_points_from_mask(mask)
        
        if grid_size > 0:
            grid_queries = get_grid_queries(
                grid_size=grid_size,
                depths=self.depth_data,
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics
            )
            query_points = torch.cat([query_points, grid_queries], dim=1)
            n_mask_points = query_points.shape[1] - grid_queries.shape[1]
        else:
            n_mask_points = query_points.shape[1]
        
        H, W = self.video_data.shape[2:]
        self.tapip3d_model.set_image_size((H, W))
        
        coords, visibs = inference(
            model=self.tapip3d_model,
            video=self.video_data,
            depths=self.depth_data,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            query_point=query_points.squeeze(0),
            num_iters=num_iters,
            grid_size=0,
            vis_threshold=vis_threshold
        )
        
        if grid_size > 0:
            coords = coords[:, :n_mask_points]
            visibs = visibs[:, :n_mask_points]
        
        print(f"TAPIP3D跟踪完成，获得 {coords.shape[1]} 个点")
        return coords, visibs
    
    def _generate_query_points_from_mask(self, mask: torch.Tensor, num_points: int = 150) -> torch.Tensor:
        mask_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(mask_indices) == 0:
            raise ValueError("mask中没有正像素")
        
        if len(mask_indices) > num_points:
            sampled_indices = torch.randperm(len(mask_indices))[:num_points]
            mask_indices = mask_indices[sampled_indices]
        
        y_coords = mask_indices[:, 0].float()
        x_coords = mask_indices[:, 1].float()
        
        query_coords_2d = torch.stack([x_coords, y_coords], dim=1)
        
        depth_frame = self.depth_data[0]
        intrinsic_frame = self.intrinsics[0]
        extrinsic_frame = self.extrinsics[0]
        depths = depth_frame[mask_indices[:, 0], mask_indices[:, 1]]
        
        inv_intrinsic = torch.linalg.inv(intrinsic_frame)
        inv_extrinsic = torch.linalg.inv(extrinsic_frame)
        query_coords_homo = torch.cat([query_coords_2d, torch.ones(len(query_coords_2d), 1, device=self.device)], dim=1)
        
        camera_coords = torch.einsum('ij,nj->ni', inv_intrinsic, query_coords_homo)
        camera_coords = camera_coords * depths.unsqueeze(1)
        camera_coords_homo = torch.cat([camera_coords, torch.ones(len(camera_coords), 1, device=self.device)], dim=1)
        
        world_coords = torch.einsum('ij,nj->ni', inv_extrinsic, camera_coords_homo)[:, :3]
        
        query_points = torch.cat([
            torch.zeros(len(world_coords), 1, device=self.device),
            world_coords
        ], dim=1).unsqueeze(0)
        
        return query_points
    
    def save_results(self, coords: torch.Tensor, visibs: torch.Tensor, 
                    mask: torch.Tensor, target_id: int, output_dir: str = "./results",
                    detection_info: Optional[Dict] = None):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        coords_path = Path(output_dir) / f"3d_tracks_target_{target_id}.npy"
        visibs_path = Path(output_dir) / f"visibility_target_{target_id}.npy"
        mask_path = Path(output_dir) / f"mask_target_{target_id}.png"
        
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        visibs_np = visibs.cpu().numpy() if torch.is_tensor(visibs) else visibs
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        
        np.save(coords_path, coords_np)
        np.save(visibs_path, visibs_np)
        
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_path)
        
        # 保存完整结果，包括检测信息
        save_dict = {
            'video': (self.video_data.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)[:1],
            'depths': self.depth_data.cpu().numpy()[:1],
            'intrinsics': self.intrinsics.cpu().numpy()[:1],
            'extrinsics': self.extrinsics.cpu().numpy()[:1],
            'coords': coords_np,
            'visibs': visibs_np,
            'mask': mask_np
        }
        
        if detection_info:
            save_dict['detection_info'] = detection_info
        
        result_path = Path(output_dir) / f"complete_result_target_{target_id}.npz"
        np.savez(result_path, **save_dict)

        print(f"结果已保存到: {result_path}")
        return result_path
    
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="使用YOLO+SAM2+TAPIP3D进行自动3D跟踪")
    parser.add_argument("--input", required=True, help="输入视频(.mp4)或npz文件路径")
    parser.add_argument("--sam2_checkpoint", required=True, help="SAM2模型checkpoint路径")
    parser.add_argument("--sam2_config", required=True, help="SAM2配置文件路径")
    parser.add_argument("--tapip3d_checkpoint", required=True, help="TAPIP3D模型checkpoint路径")
    parser.add_argument("--yolo_model", default="/data/jingjing/pretrained-models/yolo/yolov8n.pt", help="YOLO模型路径或名称")
    parser.add_argument("--output_dir", default="./yolo_sam2_tapip3d_results", help="输出目录")
    parser.add_argument("--target_resolution", type=int, nargs=2, help="目标分辨率 (H W)")
    parser.add_argument("--num_targets", type=int, default=None, help="要跟踪的目标数量，None表示跟踪所有检测到的对象")
    parser.add_argument("--grid_size", type=int, default=0, help="TAPIP3D额外的网格点数量")
    parser.add_argument("--num_iters", type=int, default=6, help="TAPIP3D跟踪迭代次数")
    parser.add_argument("--device", default="auto", help="设备: 'cuda', 'cpu', 或 'auto'")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--yolo_classes", type=int, nargs='+', help="要检测的YOLO类别ID，例如: 0 (person)")
    parser.add_argument("--erode_mask", action='store_true', help="是否腐蚀mask以获得更精确的边界")
    parser.add_argument("--visualize", action='store_true', help="是否可视化检测和分割结果")
    
    args = parser.parse_args()
    
    setup_logger()

    # 初始化集成系统
    integrator = ImprovedSAM2TAPIP3DIntegration(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        tapip3d_checkpoint=args.tapip3d_checkpoint,
        yolo_model=args.yolo_model,
        device=args.device
    )
    
    # 加载数据
    target_resolution = tuple(args.target_resolution) if args.target_resolution else None
    integrator.load_data(args.input, target_resolution)
    
    # 准备SAM2视频
    temp_dir = integrator.prepare_sam2_video()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 使用YOLO检测对象
    print("\n" + "="*60)
    print("使用YOLO检测对象...")
    print("="*60)
    detections = integrator.detect_objects_yolo(
        frame_idx=0, 
        conf_threshold=args.yolo_conf,
        class_filter=args.yolo_classes
    )
    
    if len(detections) == 0:
        print("错误: 没有检测到任何对象！")
        return 1
    
    # 可视化检测结果
    if args.visualize:
        integrator.visualize_detections(
            frame_idx=0, 
            detections=detections,
            save_path=Path(args.output_dir) / "yolo_detections.png"
        )
    
    # 确定要处理的目标数量
    num_targets_to_process = args.num_targets if args.num_targets else len(detections)
    num_targets_to_process = min(num_targets_to_process, len(detections))
    
    print(f"\n将处理 {num_targets_to_process}/{len(detections)} 个检测到的对象")
    
    # 对每个检测到的对象进行分割和跟踪
    results = []
    for target_idx in range(num_targets_to_process):
        detection = detections[target_idx]
        target_id = target_idx + 1
        
        print(f"\n{'='*60}")
        print(f"处理目标 {target_id}/{num_targets_to_process}")
        print(f"类别: {detection['class_name']}, 置信度: {detection['confidence']:.2f}")
        print(f"Bounding Box: {detection['bbox']}")
        print(f"{'='*60}")
        
        # 使用bbox生成mask
        mask = integrator.auto_segment_with_bbox(
            frame_idx=0, 
            bbox=detection['bbox'],
            target_id=target_id
        )
        
        # 可选：腐蚀mask
        if args.erode_mask:
            kernel_size = 5
            mask_tensor = mask.float().unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask_tensor.device)
            mask_tensor = F.conv2d(F.pad(mask_tensor, (2, 2, 2, 2)), kernel)
            mask = (mask_tensor == kernel_size * kernel_size).float().squeeze()
            print("Mask已腐蚀")
        
        # 可视化mask
        if args.visualize:
            integrator.visualize_mask(
                frame_idx=0,
                mask=mask,
                bbox=detection['bbox'],
                title=f"目标 {target_id} - {detection['class_name']}",
                save_path=Path(args.output_dir) / f"mask_target_{target_id}.png"
            )
        
        # 使用TAPIP3D跟踪
        coords, visibs = integrator.track_with_tapip3d(
            mask=mask,
            grid_size=args.grid_size,
            num_iters=args.num_iters,
            vis_threshold=0.9
        )
        
        # 保存结果
        result_path = integrator.save_results(
            coords, visibs, mask, target_id, args.output_dir,
            detection_info=detection
        )
        results.append(result_path)
    
    # 清理临时目录
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"\n已清理临时目录: {temp_dir}")
    
    integrator.cleanup()
    
    print(f"\n{'='*60}")
    print(f"全部完成！处理了 {len(results)} 个目标")
    print(f"结果保存在: {args.output_dir}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())