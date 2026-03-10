import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
import tempfile
import shutil
import torch.nn.functional as F

# GroundingDINO相关导入
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
import supervision as sv

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


class GroundingDINODetector:
    """GroundingDINO检测器封装类"""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(config_path, checkpoint_path)
        
    def _load_model(self, config_path: str, checkpoint_path: str):
        """加载GroundingDINO模型"""
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model = model.to(self.device)
        return model
    
    def detect(self, image: np.ndarray, text_prompt: str, 
               box_threshold: float = 0.35, text_threshold: float = 0.25) -> List[Dict]:
        """
        使用文本提示检测物体
        
        Args:
            image: RGB图像 (H, W, 3) uint8格式
            text_prompt: 文本描述，例如 "cup . teddy bear" (用 . 分隔多个类别)
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
        
        Returns:
            检测结果列表，每个元素包含bbox、confidence、class_name
        """
        # 图像预处理
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(image)
        image_transformed, _ = transform(image_pil, None)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                image_transformed.unsqueeze(0).to(self.device),
                captions=[text_prompt]
            )
        
        # 解析结果
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # 过滤低置信度的检测
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # 获取文本短语
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(text_prompt)
        
        pred_phrases = []
        pred_boxes = []
        pred_scores = []
        
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            if pred_phrase:  # 只添加有效的检测
                pred_phrases.append(pred_phrase)
                pred_boxes.append(box.tolist())
                pred_scores.append(logit.max().item())
        
        # 转换box格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        H, W = image.shape[:2]
        detections = []
        for phrase, box, score in zip(pred_phrases, pred_boxes, pred_scores):
            cx, cy, w, h = box
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(score),
                'class_name': phrase
            })
        
        return detections


class ImprovedSAM2TAPIP3DIntegration:
    def __init__(self, sam2_checkpoint: str, sam2_config: str, tapip3d_checkpoint: str, 
                 grounding_dino_config: str, grounding_dino_checkpoint: str, 
                 device: str = "auto"):
        self.device = self._setup_device(device)
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.tapip3d_checkpoint = tapip3d_checkpoint
        
        self._init_sam2()
        self._init_tapip3d()
        self._init_grounding_dino(grounding_dino_config, grounding_dino_checkpoint)
        
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
    
    def _init_grounding_dino(self, config_path: str, checkpoint_path: str):
        """初始化GroundingDINO模型"""
        self.grounding_dino_detector = GroundingDINODetector(
            config_path, checkpoint_path, self.device
        )
        print(f"GroundingDINO模型已加载到 {self.device}")
    
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
    
    def detect_objects_grounding_dino(self, frame_idx: int = 0, text_prompt: str = "cup . teddy bear",
                                     box_threshold: float = 0.35, text_threshold: float = 0.25) -> List[Dict]:
        """
        使用GroundingDINO检测第一帧中的对象
        
        Args:
            frame_idx: 帧索引
            text_prompt: 文本描述，例如 "cup . teddy bear" (用 . 分隔多个类别)
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
        
        Returns:
            检测结果列表，每个元素包含bbox、confidence、class_name
        """
        frame = self.video_data[frame_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        # GroundingDINO推理
        detections = self.grounding_dino_detector.detect(
            frame, text_prompt, box_threshold, text_threshold
        )
        
        print(f"在第{frame_idx}帧检测到 {len(detections)} 个对象")
        print(f"文本提示: '{text_prompt}'")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det['class_name']}: {det['confidence']:.3f}")
        
        return detections
    
    def visualize_detections(self, frame_idx: int, detections: List[Dict], 
                            save_path: Optional[str] = None):
        """可视化检测结果"""
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
        
        plt.title(f"GroundingDINO检测结果 - 第{frame_idx}帧 (共{len(detections)}个对象)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"检测结果已保存到: {save_path}")
        
        plt.close()
    
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
        
        plt.close()

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

        print(coords.shape)
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
    parser = argparse.ArgumentParser(description="使用GroundingDINO+SAM2+TAPIP3D进行自动3D跟踪")
    parser.add_argument("--input", required=True, help="输入视频(.mp4)或npz文件路径")
    parser.add_argument("--sam2_checkpoint", required=True, help="SAM2模型checkpoint路径")
    parser.add_argument("--sam2_config", required=True, help="SAM2配置文件路径")
    parser.add_argument("--tapip3d_checkpoint", required=True, help="TAPIP3D模型checkpoint路径")
    parser.add_argument("--grounding_dino_config", required=True, help="GroundingDINO配置文件路径")
    parser.add_argument("--grounding_dino_checkpoint", required=True, help="GroundingDINO模型checkpoint路径")
    parser.add_argument("--output_dir", default="./grounding_dino_sam2_tapip3d_results", help="输出目录")
    parser.add_argument("--target_resolution", type=int, nargs=2, help="目标分辨率 (H W)")
    parser.add_argument("--num_targets", type=int, default=None, help="要跟踪的目标数量，None表示跟踪所有检测到的对象")
    parser.add_argument("--grid_size", type=int, default=0, help="TAPIP3D额外的网格点数量")
    parser.add_argument("--num_iters", type=int, default=6, help="TAPIP3D跟踪迭代次数")
    parser.add_argument("--device", default="auto", help="设备: 'cuda', 'cpu', 或 'auto'")
    parser.add_argument("--text_prompt", default="cup . teddy bear", help="GroundingDINO文本提示，例如: 'cup . teddy bear'")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="GroundingDINO边界框置信度阈值")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO文本匹配阈值")
    parser.add_argument("--erode_mask", action='store_true', help="是否腐蚀mask以获得更精确的边界")
    parser.add_argument("--visualize", action='store_true', help="是否可视化检测和分割结果")
    
    args = parser.parse_args()
    
    setup_logger()

    # 初始化集成系统
    integrator = ImprovedSAM2TAPIP3DIntegration(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        tapip3d_checkpoint=args.tapip3d_checkpoint,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        device=args.device
    )
    
    # 加载数据
    target_resolution = tuple(args.target_resolution) if args.target_resolution else None
    integrator.load_data(args.input, target_resolution)
    
    # 准备SAM2视频
    temp_dir = integrator.prepare_sam2_video()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 使用GroundingDINO检测对象
    print("\n" + "="*60)
    print("使用GroundingDINO检测对象...")
    print("="*60)
    detections = integrator.detect_objects_grounding_dino(
        frame_idx=0, 
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    
    if len(detections) == 0:
        print("错误: 没有检测到任何对象！")
        print("提示: 尝试降低 --box_threshold 或 --text_threshold")
        return 1
    
    # 可视化检测结果
    if args.visualize:
        integrator.visualize_detections(
            frame_idx=0, 
            detections=detections,
            save_path=Path(args.output_dir) / "grounding_dino_detections.png"
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