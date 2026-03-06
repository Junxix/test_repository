import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
import sys
sys.path.append('/home/ubuntu/git/jingjing/testspace/TAPIP3D/')

from utils.inference_utils import load_model, inference_with_mask, get_grid_queries, inference
sys.path.append('/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/sam2/')

from sam2.build_sam import build_sam2_video_predictor

# SigLIP
from transformers import SiglipModel, SiglipImageProcessor

from policy import RISE
from eval_agent import Agent
from util.constants import (
    IMG_MEAN, IMG_STD, TRANS_MIN, TRANS_MAX, MAX_GRIPPER_WIDTH,
    WORKSPACE_MIN, WORKSPACE_MAX, SAFE_WORKSPACE_MIN, SAFE_WORKSPACE_MAX,
    SAFE_EPS, GRIPPER_THRESHOLD, TRACK_MIN, TRACK_MAX
)
# from dataset.constants import REL_TRANS_MAX, REL_GRIPPER_MAX
from util.training import set_seed
from dataset.projector import Projector
from util.ensemble import EnsembleBuffer
from util.transformation import rotation_transform

from online_tracker import (
    create_point_cloud, 
    create_batch, 
    create_input, 
    unnormalize_action, 
    rot_diff, 
    discretize_rotation
)

default_args = edict({
    "ckpt": None,
    "calib": "calib/",
    "human_demo_path": "/data/jingjing/data/context/realdata_sampled_mismatch/train/task_0103_user_0555_scene_0021_cfg_0001_BEFORE_task_0103_user_0555_scene_0028_cfg_0001_AFTER",
    "num_action": 20,
    "num_history": 5,
    "num_inference_step": 16,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_steps": 300,
    "seed": 233,
    "vis": False,
    "discretize_rotation": True,
    "ensemble_mode": "act",
    "sam2_checkpoint": "/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/sam2/sam2.1_hiera_large.pt",
    "sam2_config": "/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/sam2/sam2.1_hiera_l.yaml",
    "tapip3d_checkpoint": "/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/TAPIP3D/checkpoints/tapip3d_final.pth",
    "siglip_model_path": "/data/pretrained-weights/siglip-so400m-patch14-384",
    "tracking_window_size": 16,
    "tracking_step_size": 8,
    "num_targets": 4,
    "points_per_target": 10,
    "siglip_image_size": 896,
    "siglip_patch_size": 14
})


class HumanTrackLoader:
    """加载预存的human demonstration tracks"""
    
    def __init__(self, demo_path, cam_id, num_targets=2, points_per_target=10):
        self.demo_path = demo_path
        self.cam_id = cam_id
        self.num_targets = num_targets
        self.points_per_target = points_per_target
        
        # # 读取human.json获取robot_start_idx
        # human_json_path = os.path.join(demo_path, "human.json")
        # if not os.path.exists(human_json_path):
        #     raise FileNotFoundError(f"human.json not found in {demo_path}")
        
        # with open(human_json_path, "r") as f:
        #     human_data = json.load(f)
        #     self.robot_start_idx = int(human_data.get("robot_start_idx", 0))
        
        # print(f"Human demo robot_start_idx: {self.robot_start_idx}")
        
        # 加载预存的human tracks和semantic features
        self._load_human_tracks()
    

    def world_to_camera_coords(self, world_coords, extrinsics):
        """
        将世界坐标系下的3D点转换为相机坐标系
        
        Args:
            world_coords: (T, N, 3) or (N, 3) - 世界坐标系下的点
            extrinsics: (4, 4) - 相机外参矩阵 (从相机到世界的变换)
                注意：extrinsics 表示相机在世界坐标系中的位姿
                world = extrinsics @ camera
                因此：camera = inv(extrinsics) @ world
        
        Returns:
            camera_coords: 相机坐标系下的点，shape与输入相同
        """
        original_shape = world_coords.shape
        
        # Flatten to (M, 3) where M = T*N or N
        if world_coords.ndim == 3:
            T, N, _ = world_coords.shape
            world_coords_flat = world_coords.reshape(-1, 3)
        else:
            world_coords_flat = world_coords
        
        # 转换为齐次坐标 (M, 4)
        ones = np.ones((world_coords_flat.shape[0], 1), dtype=world_coords_flat.dtype)
        world_coords_homo = np.concatenate([world_coords_flat, ones], axis=1)
        
        # ===== 关键修改：需要使用外参的逆矩阵 =====
        # camera = inv(extrinsics) @ world
        inv_extrinsics = np.linalg.inv(extrinsics)
        camera_coords_homo = (inv_extrinsics @ world_coords_homo.T).T
        # ===== 修改结束 =====
        
        # 转回3D坐标
        camera_coords = camera_coords_homo[:, :3]
        
        # Reshape回原始形状
        if len(original_shape) == 3:
            camera_coords = camera_coords.reshape(original_shape)
        
        return camera_coords
    
    def _load_human_tracks(self):
        """加载human tracks和semantic features（与realworld.py一致）"""
        cam_path = self.demo_path
        
        # Human tracks目录
        human_tracks_dir = os.path.join(cam_path, "before_sam2_tapip3d_results_offline")
        human_semantic_dir = os.path.join(cam_path, "human_siglip")
        
        if not os.path.exists(human_tracks_dir):
            raise FileNotFoundError(f"Human tracks directory not found: {human_tracks_dir}")
        if not os.path.exists(human_semantic_dir):
            raise FileNotFoundError(f"Human semantic directory not found: {human_semantic_dir}")
        
        human_point_tracks = []
        human_semantic_features = []
        
        extrinsics = np.array([[ 1., 0., 0.,  0.],
                            [0.,  1., 0., 0.],
                            [0. ,  0.,1.,  0.],
                            [ 0., 0., 0., 1.]])
                
        for target_idx in range(1, self.num_targets + 1):
            # Load tracks (world coords)
            human_target_path = os.path.join(human_tracks_dir, f"3d_tracks_target_before_{target_idx}.npy")
            if not os.path.exists(human_target_path):
                raise FileNotFoundError(f"Human target {target_idx} track file not found")
            
            human_pred_tracks = np.load(human_target_path)
            if self.num_targets == 1 and human_pred_tracks.ndim == 4:
                human_pred_tracks = human_pred_tracks[0]
            
            if human_pred_tracks.shape[1] < self.points_per_target:
                raise ValueError(f"Not enough points in human target_{target_idx}")
            
            # Convert to camera coords
            human_pred_tracks = self.world_to_camera_coords(human_pred_tracks, extrinsics)
            
            # FPS sampling
            human_first_frame_points = human_pred_tracks[0]
            human_point_indices = self._fps_sampling_3d(human_first_frame_points, self.points_per_target)
            human_tracks = human_pred_tracks[:, human_point_indices, :]  # (T, num_points, 3)
            
            # # === Compute absolute version ===
            # human_tracks_abs_normalized = (human_tracks - TRACK_MIN) / (TRACK_MAX - TRACK_MIN) * 2 - 1
            # human_tracks = np.clip(human_tracks_abs_normalized, -1.0, 1.0)
            

            # Load semantics - 每个target是单独的.npy文件
            human_semantic_path = os.path.join(human_semantic_dir, f"target_{target_idx}.npy")
            if not os.path.exists(human_semantic_path):
                raise FileNotFoundError(f"Human semantic file not found: {human_semantic_path}")
            human_semantic = np.load(human_semantic_path)
            
            human_point_tracks.append(human_tracks)
            human_semantic_features.append(human_semantic)
        
        # Concatenate all targets - 与realworld.py完全一致
        self.human_tracks = np.concatenate(human_point_tracks, axis=1)
        self.human_semantics = np.stack(human_semantic_features, axis=1)  # 注意是axis=1
        self.human_semantics = self.human_semantics[0,:,:]
        
        print(f"Loaded human tracks: {self.human_tracks.shape}")
        print(f"Loaded human semantics: {self.human_semantics.shape}")
    
    def _fps_sampling_3d(self, points, n_samples):
        """3D最远点采样"""
        N = points.shape[0]
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(N) * 1e10
        
        # 随机选择第一个点
        farthest = np.random.randint(0, N)
        
        for i in range(n_samples):
            sampled_indices[i] = farthest
            centroid = points[farthest]
            
            # 计算所有点到当前点的欧氏距离(3D)
            dist = np.sum((points - centroid) ** 2, axis=1)
            
            # 更新每个点到已选点集的最小距离
            distances = np.minimum(distances, dist)
            
            # 选择距离最远的点
            farthest = np.argmax(distances)
        
        return sampled_indices
    
    def get_human_tracks(self):
        """返回完整的human tracks"""
        return self.human_tracks
    
    def get_human_semantics(self):
        """返回human semantic features"""
        return self.human_semantics


class RobotTracker:
    """实时跟踪robot执行过程中的物体，并提取semantic features"""
    
    def __init__(self, sam2_checkpoint, sam2_config, tapip3d_checkpoint,
                 siglip_model_path, window_size=16, step_size=8, device="cuda", 
                 num_targets=2, image_size=896, patch_size=14):
        self.device = device
        self.window_size = window_size
        self.step_size = step_size
        self.num_targets = num_targets
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        
        # 初始化SAM2和TAPIP3D
        self._init_sam2(sam2_checkpoint, sam2_config)
        self._init_tapip3d(tapip3d_checkpoint)
        
        # 初始化SigLIP
        self._init_siglip(siglip_model_path)
        
        from collections import deque
        self.frame_buffer = deque(maxlen=window_size)
        self.depth_buffer = deque(maxlen=window_size)
        self.intrinsics_buffer = deque(maxlen=window_size)
        self.extrinsics_buffer = deque(maxlen=window_size)
        
        self.global_coords = []  # 存储所有目标的3D坐标
        self.global_frame_count = 0
        
        self.targets = {}  # target_id -> target_info
        self.robot_semantics = None  # 存储robot的semantic features (num_targets, feature_dim)
        self.initialized = False
    
    def _init_sam2(self, checkpoint, config):
        """初始化SAM2"""
        self.sam2_predictor = build_sam2_video_predictor(
            config, checkpoint, device=self.device
        )
    
    def _init_tapip3d(self, checkpoint):
        """初始化TAPIP3D"""
        self.tapip3d_model = load_model(checkpoint)
        self.tapip3d_model.to(self.device)
        self.tapip3d_model.eval()
        
        if hasattr(self.tapip3d_model, "set_eval_mode"):
            self.tapip3d_model.set_eval_mode("raw")
    
    def _init_siglip(self, model_path):
        """初始化SigLIP模型"""
        print("Loading SigLIP model...")
        self.siglip_model = SiglipModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.siglip_processor = SiglipImageProcessor.from_pretrained(model_path)
        self.siglip_vision_model = self.siglip_model.vision_model
        self.siglip_vision_model.to(self.device)
        self.siglip_vision_model.eval()
        print("SigLIP model loaded!")
    
    def _extract_siglip_features(self, image, mask, image_path):
        """
        从图像和mask提取SigLIP semantic features
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            mask: numpy array (H, W), binary mask
            
        Returns:
            semantic_feature: numpy array (1, feature_dim)
        """
        # if isinstance(image, np.ndarray):
        #     image = Image.fromarray(image)
        image = Image.open(image_path).convert('RGB')

        # Resize到896x896并归一化
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # SigLIP归一化
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image_normalized = (image_array - mean) / std
        
        # 转换为tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
        pixel_values = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.siglip_vision_model(
                pixel_values=pixel_values, 
                output_hidden_states=True, 
                interpolate_pos_encoding=True
            )
            features = outputs.last_hidden_state
        
        # 提取patch tokens
        patch_tokens = features.squeeze(0)  # [num_patches, hidden_dim]
        
        # Resize mask到896x896
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_array = np.array(mask_resized)
        mask_binary = (mask_array > 0).astype(np.float32)
        
        # 计算哪些patches被mask覆盖
        selected_patches = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y_start = row * self.patch_size
                y_end = (row + 1) * self.patch_size
                x_start = col * self.patch_size
                x_end = (col + 1) * self.patch_size
                
                patch_mask = mask_binary[y_start:y_end, x_start:x_end]
                coverage = patch_mask.sum() / (self.patch_size * self.patch_size)
                
                if coverage >= 0.5:  # 50%以上覆盖
                    patch_idx = row * self.grid_size + col
                    selected_patches.append(patch_idx)
        
        # 计算平均特征
        if len(selected_patches) > 0:
            selected_features = patch_tokens[selected_patches]
            reference_feature = selected_features.mean(dim=0, keepdim=True)
            return reference_feature.cpu().numpy()
        else:
            raise ValueError("error")
    
    def interactive_segment_objects(self, frame, depth, intrinsics, extrinsics, 
                                   save_dir="./robot_masks"):
        """交互式分割多个物体并提取semantic features"""
        import tempfile
        import shutil
        from pathlib import Path
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存原始图像
        original_img_path = os.path.join(save_dir, "robot_frame.png")
        Image.fromarray(frame).save(original_img_path)
        print(f"Robot frame saved to: {original_img_path}")
        
        # 存储所有目标的semantic features
        all_semantic_features = []
        
        for target_id in range(1, self.num_targets + 1):
            print(f"\n=== Segmenting robot object {target_id} ===")
            
            temp_dir = tempfile.mkdtemp()
            try:
                frame_path = Path(temp_dir) / "000000.jpg"
                Image.fromarray(frame).save(frame_path)
                
                # 为每个物体创建独立的inference_state
                inference_state = self.sam2_predictor.init_state(video_path=temp_dir)
                
                points = []
                labels = []
                mask = None
                
                plt.figure(figsize=(12, 8))
                plt.title(f"Select points for robot object {target_id} (left=positive, right=negative, q=finish, z=undo)")
                plt.imshow(frame)
                
                def on_click(event):
                    nonlocal points, labels, mask
                    
                    if event.xdata is None or event.ydata is None:
                        return
                    
                    x, y = int(event.xdata), int(event.ydata)
                    
                    if event.button in [1, 3]:
                        label = 1 if event.button == 1 else 0
                        points.append([x, y])
                        labels.append(label)
                        print(f"  Added point ({x}, {y}) label: {'positive' if label == 1 else 'negative'}")
                        
                        if len(points) > 0:
                            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=1,
                                points=np.array(points),
                                labels=np.array(labels)
                            )
                            
                            # 清除之前的可视化
                            ax = plt.gca()
                            images = ax.images
                            if len(images) > 1:
                                for img in images[1:]:
                                    img.remove()
                            for collection in ax.collections:
                                collection.remove()
                            
                            if len(out_obj_ids) > 0:
                                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                
                                # 显示选择的点
                                pos_points = np.array(points)[np.array(labels) == 1]
                                neg_points = np.array(points)[np.array(labels) == 0]
                                
                                if len(pos_points) > 0:
                                    ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                                            color='green', marker='*', s=200, 
                                            edgecolor='white', linewidth=1.25)
                                if len(neg_points) > 0:
                                    ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                                            color='red', marker='*', s=200, 
                                            edgecolor='white', linewidth=1.25)
                                
                                color = np.array([*plt.get_cmap("tab10")(target_id)[:3], 0.6])
                                h, w = mask.shape
                                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                                ax.imshow(mask_image, alpha=0.6)
                                
                            plt.draw()
                
                def on_key(event):
                    nonlocal points, labels, mask
                    
                    if event.key == 'z' and len(points) > 0:
                        removed_point = points.pop()
                        removed_label = labels.pop()
                        print(f"  Undoing point: {removed_point}")
                        
                        ax = plt.gca()
                        images = ax.images
                        if len(images) > 1:
                            for img in images[1:]:
                                img.remove()
                        for collection in ax.collections:
                            collection.remove()
                        
                        if len(points) > 0:
                            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=1,
                                points=np.array(points),
                                labels=np.array(labels)
                            )
                            
                            if len(out_obj_ids) > 0:
                                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                                if len(mask.shape) == 3:
                                    mask = mask[0]
                                
                                pos_points = np.array(points)[np.array(labels) == 1]
                                neg_points = np.array(points)[np.array(labels) == 0]
                                
                                if len(pos_points) > 0:
                                    ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                                            color='green', marker='*', s=200, 
                                            edgecolor='white', linewidth=1.25)
                                if len(neg_points) > 0:
                                    ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                                            color='red', marker='*', s=200, 
                                            edgecolor='white', linewidth=1.25)
                                
                                color = np.array([*plt.get_cmap("tab10")(target_id)[:3], 0.6])
                                h, w = mask.shape
                                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                                ax.imshow(mask_image, alpha=0.6)
                        else:
                            mask = None
                            
                        plt.draw()
                        
                    elif event.key == 'q':
                        plt.close()
                
                canvas = plt.gcf().canvas
                canvas.mpl_connect('button_press_event', on_click)
                canvas.mpl_connect('key_press_event', on_key)
                
                plt.show()
                
                if mask is None:
                    print(f"Robot object {target_id} segmentation failed")
                    return False
                
                # 保存mask
                print(f"Robot object {target_id} segmentation completed, mask shape: {mask.shape}")
                
                mask_filename = f"robot_object_{target_id}_mask.npy"
                mask_path = os.path.join(save_dir, mask_filename)
                np.save(mask_path, mask.astype(np.uint8))
                print(f"Robot object {target_id} mask saved to: {mask_path}")
                
                # **提取semantic features**
                print(f"Extracting semantic features for robot object {target_id}...")
                semantic_feature = self._extract_siglip_features(frame, mask, original_img_path)
                all_semantic_features.append(semantic_feature)
                print(f"Semantic feature shape: {semantic_feature.shape}")
                
                # 生成query points
                query_points = self._generate_query_points_from_mask(
                    torch.from_numpy(mask).to(self.device),
                    torch.from_numpy(depth).to(self.device),
                    torch.from_numpy(intrinsics).to(self.device),
                    torch.from_numpy(extrinsics).to(self.device),
                    num_points=10
                )
                
                self.targets[target_id] = {
                    'mask': mask,
                    'query_points': query_points,
                    'initialized': True,
                    'last_coords': None,
                    'last_visibs': None
                }
                
                print(f"Robot object {target_id} initialized with {query_points.shape[1]} query points")
                
            finally:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
        
        # 合并所有semantic features
        if len(all_semantic_features) == self.num_targets:
            self.robot_semantics = np.concatenate(all_semantic_features, axis=0)  # (num_targets, feature_dim)
            print(f"\nRobot semantic features shape: {self.robot_semantics.shape}")
            
            # 保存semantic features
            semantic_save_path = os.path.join(save_dir, "robot_semantic_features.npy")
            np.save(semantic_save_path, self.robot_semantics)
            print(f"Robot semantic features saved to: {semantic_save_path}")
        else:
            raise ValueError("Not all robot object semantic features were extracted successfully.")
        
        self.initialized = True
        return True
    

    def erode_mask(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        对二值mask进行腐蚀操作
        
        参数:
            mask: 二值mask, shape为(H, W), torch.Tensor
            kernel_size: 腐蚀核大小,必须是奇数
        
        返回:
            腐蚀后的mask
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保是奇数
        
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        
        padding = kernel_size // 2
        convolved = F.conv2d(mask_float, kernel, padding=padding)
        
        threshold = kernel_size * kernel_size
        eroded_mask = (convolved >= threshold).squeeze(0).squeeze(0)
        
        return eroded_mask
    
    def _generate_query_points_from_mask(self, mask, depth, intrinsics, extrinsics, num_points=20):
        """从mask生成query points，使用FPS采样"""
        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()
        mask = self.erode_mask(mask, kernel_size=10)
        mask_indices = torch.nonzero(mask, as_tuple=False)
        
        # 首先back-project所有mask点到3D
        y_coords = mask_indices[:, 0].float()
        x_coords = mask_indices[:, 1].float()
        
        query_coords_2d = torch.stack([x_coords, y_coords], dim=1)
        depths = depth[mask_indices[:, 0], mask_indices[:, 1]]
        
        # Back-project到3D
        inv_intrinsic = torch.linalg.inv(intrinsics)
        inv_extrinsic = torch.linalg.inv(extrinsics)
        
        query_coords_homo = torch.cat([query_coords_2d, torch.ones(len(query_coords_2d), 1, device=self.device)], dim=1)
        camera_coords = torch.einsum('ij,nj->ni', inv_intrinsic, query_coords_homo)
        camera_coords = camera_coords * depths.unsqueeze(1)
        camera_coords_homo = torch.cat([camera_coords, torch.ones(len(camera_coords), 1, device=self.device)], dim=1)
        
        world_coords = torch.einsum('ij,nj->ni', inv_extrinsic, camera_coords_homo)[:, :3]
        
        # 使用FPS采样到num_points个点
        if len(world_coords) > num_points:
            world_coords_np = world_coords.cpu().numpy()
            sampled_indices = self._fps_sampling_3d(world_coords_np, num_points)
            world_coords = world_coords[sampled_indices]
        
        query_points = torch.cat([
            torch.zeros(len(world_coords), 1, device=self.device),
            world_coords
        ], dim=1).unsqueeze(0)
        
        return query_points
    
    def add_frame(self, frame, depth, intrinsics, extrinsics):
        """添加新帧并可能触发推理"""
        self.frame_buffer.append(frame)
        self.depth_buffer.append(depth)
        self.intrinsics_buffer.append(intrinsics)
        self.extrinsics_buffer.append(extrinsics)
        
        self.global_frame_count += 1
        
        if (self.global_frame_count % self.step_size == 0 and 
            len(self.frame_buffer) >= self.window_size and
            self.initialized) or self.global_frame_count == 1:
            
            return self._run_inference()
        
        return {}
    
    def _run_inference(self):
        """运行TAPIP3D推理"""
        results = {}
        
        window_frames = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 
            for frame in self.frame_buffer
        ]).to(self.device)
        
        window_depths = torch.stack([
            torch.from_numpy(depth).float() 
            for depth in self.depth_buffer
        ]).to(self.device)
        
        window_intrinsics = torch.stack([
            torch.from_numpy(intrinsics).float() 
            for intrinsics in self.intrinsics_buffer
        ]).to(self.device)
        
        window_extrinsics = torch.stack([
            torch.from_numpy(extrinsics).float() 
            for extrinsics in self.extrinsics_buffer
        ]).to(self.device)
        
        H, W = window_frames.shape[2:]
        self.tapip3d_model.set_image_size((H, W))
        
        initialized_targets = {}
        all_query_points = []
        target_point_counts = []
        target_ids_list = []
        
        if window_frames.shape[0] == 1:
            window_frames = window_frames.repeat(16, 1, 1, 1)
            window_depths = window_depths.repeat(16, 1, 1)
            window_intrinsics = window_intrinsics.repeat(16, 1, 1)
            window_extrinsics = window_extrinsics.repeat(16, 1, 1)
        
        for target_id, target_info in self.targets.items():
            if not target_info['initialized']:
                continue
            
            query_points = self._get_query_points_for_target(target_id, window_frames[0])
            query_points_squeezed = query_points.squeeze(0)
            
            initialized_targets[target_id] = target_info
            all_query_points.append(query_points_squeezed)
            target_point_counts.append(query_points_squeezed.shape[0])
            target_ids_list.append(target_id)
        
        if not initialized_targets:
            return results
        
        combined_query_points = torch.cat(all_query_points, dim=0)
        
        coords, visibs = inference(
            model=self.tapip3d_model,
            video=window_frames,
            depths=window_depths,
            intrinsics=window_intrinsics,
            extrinsics=window_extrinsics,
            query_point=combined_query_points,
            num_iters=6,
            grid_size=0,
            vis_threshold=0.9
        )
        
        start_idx = 0
        for i, target_id in enumerate(target_ids_list):
            point_count = target_point_counts[i]
            end_idx = start_idx + point_count
            
            target_coords = coords[:, start_idx:end_idx, :]
            target_visibs = visibs[:, start_idx:end_idx]
            
            target_info = initialized_targets[target_id]
            target_info['last_coords'] = target_coords
            target_info['last_visibs'] = target_visibs
            
            self._update_global_results(target_id, target_coords, target_visibs)
            
            results[target_id] = (
                self._get_global_coords(target_id),
                self._get_global_visibs(target_id)
            )
            
            start_idx = end_idx
        
        return results
    
    def _get_query_points_for_target(self, target_id, current_frame):
        """获取目标的query points"""
        target_info = self.targets[target_id]
        
        if target_info['last_coords'] is not None and target_info['last_visibs'] is not None:
            last_coords = target_info['last_coords'][self.step_size]
            last_visibs = target_info['last_visibs'][self.step_size]
            
            visible_mask = last_visibs >= 0
            if visible_mask.sum() > 0:
                visible_coords = last_coords[visible_mask]
                
                query_points = torch.cat([
                    torch.zeros(len(visible_coords), 1, device=self.device),
                    visible_coords
                ], dim=1).unsqueeze(0)
                
                return query_points
        
        return target_info['query_points']
    
    def _update_global_results(self, target_id, coords, visibs):
        """更新全局跟踪结果"""
        while len(self.global_coords) <= target_id:
            self.global_coords.append([])
        
        start_frame = max(0, self.global_frame_count - self.window_size)
        for t in range(coords.shape[0]):
            frame_idx = start_frame + t
            
            if frame_idx < len(self.global_coords[target_id]):
                self.global_coords[target_id][frame_idx] = coords[t]
            else:
                self.global_coords[target_id].append(coords[t])
    
    def _get_global_coords(self, target_id):
        """获取全局坐标"""
        if target_id >= len(self.global_coords) or len(self.global_coords[target_id]) == 0:
            raise ValueError(f"No coordinates for target {target_id}")
        
        return torch.stack(self.global_coords[target_id])
    
    def _get_global_visibs(self, target_id):
        """获取可见性"""
        return None
    
    def get_robot_tracks_for_policy(self) -> torch.Tensor:
        all_tracks_history = []
        
        for target_id in range(1, self.num_targets + 1):
            if target_id in self.targets:
                coords = self._get_global_coords(target_id)  # (seq_len, num_points, 3)
                if coords.numel() > 0:
                    all_tracks_history.append(coords)
        
        if len(all_tracks_history) == self.num_targets:
            combined_tracks = torch.cat(all_tracks_history, dim=1)  # (seq_len, 20, 3)
            return combined_tracks
        else:
            # If not enough tracking results, return zero tensor
            return torch.zeros(1, 20, 3, device=self.device)
    
    def get_robot_semantics(self):
        """获取robot semantic features"""
        return self.robot_semantics
    
    def _fps_sampling_3d(self, points, n_samples):
        """3D FPS采样"""
        N = points.shape[0]
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(N) * 1e10
        
        farthest = np.random.randint(0, N)
        
        for i in range(n_samples):
            sampled_indices[i] = farthest
            centroid = points[farthest]
            dist = np.sum((points - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
        
        return sampled_indices




import torch
import os

# ============= 存储代码 =============
def save_tracks_and_semantics(save_dir, episode_idx, 
                                human_tracks_abs, robot_tracks_abs,
                              human_tracks_rel, robot_tracks_rel,
                              human_semantics, robot_semantics,
                              human_track_lengths=None, 
                              robot_track_lengths=None,cloud_feats=None,cloud_coords=None):
    """
    保存tracks和semantics数据
    
    Args:
        save_dir: 保存目录
        episode_idx: episode索引
        human_tracks: human轨迹数据
        robot_tracks: robot轨迹数据
        human_semantics: human语义特征
        robot_semantics: robot语义特征
        human_track_lengths: human轨迹长度(可选)
        robot_track_lengths: robot轨迹长度(可选)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {
        'human_tracks_abs': human_tracks_abs.cpu() if human_tracks_abs is not None else None,
        'robot_tracks_abs': robot_tracks_abs.cpu() if robot_tracks_abs is not None else None,
        'human_tracks_rel': human_tracks_rel.cpu() if human_tracks_rel is not None else None,
        'robot_tracks_rel': robot_tracks_rel.cpu() if robot_tracks_rel is not None else None,
        'human_semantics': human_semantics.cpu() if human_semantics is not None else None,
        'robot_semantics': robot_semantics.cpu() if robot_semantics is not None else None,
        'cloud_feats': cloud_feats.cpu() if cloud_feats is not None else None,
        'cloud_coords': cloud_coords.cpu() if cloud_coords is not None else None,
    }
    
    if human_track_lengths is not None:
        save_data['human_track_lengths'] = human_track_lengths.cpu()
    if robot_track_lengths is not None:
        save_data['robot_track_lengths'] = robot_track_lengths.cpu()
    
    save_path = os.path.join(save_dir, f'tracks_semantics_{episode_idx}.pt')
    torch.save(save_data, save_path)
    print(f"已保存到: {save_path}")


# ============= 读取代码 =============
def load_tracks_and_semantics(save_dir, episode_idx, device):
    """
    读取tracks和semantics数据
    
    Args:
        save_dir: 保存目录
        episode_idx: episode索引
        device: 目标设备
        
    Returns:
        dict: 包含所有数据的字典
    """
    load_path = os.path.join(save_dir, f'tracks_semantics_{episode_idx}.pt')
    
    if not os.path.exists(load_path):
        return None
    
    data = torch.load(load_path, map_location='cpu')
    
    # 将数据移到目标设备
    result = {}
    for key, value in data.items():
        if value is not None:
            result[key] = value.to(device)
        else:
            result[key] = None
    
    print(f"已从 {load_path} 读取数据")
    return result

def evaluate(args_override):
    # 加载参数
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # 设置设备
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track encoder配置
    track_config = {
        'input_dim': 3,
        'patch_size': 4,
        'embed_dim': 256,
        'query_dim': 256,
        'num_queries': 1,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': False,
        'use_time_embedding': True,
        'output_dim': None
    }

    # 加载policy
    print("Loading policy ...")
    policy = RISE(
        num_action=args.num_action,
        input_dim=6,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_attn_layers=4,
        dropout=args.dropout,
        track_config=track_config,
        num_targets=args.num_targets,
        num_points=10
    ).to(device)
    
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # 加载checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # 初始化agent
    agent = Agent(
        robot_ip="192.168.2.100",
        pc_ip="192.168.2.35",
        gripper_port="/dev/ttyUSB0",
        camera_serial="104122063550"
    )
    projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)
    
    # 加载human tracks
    print("Loading human demonstration tracks...")
    human_loader = HumanTrackLoader(
        demo_path=args.human_demo_path,
        cam_id=agent.camera_serial,
        num_targets=args.num_targets,
        points_per_target=args.points_per_target
    )
    
    human_tracks = torch.from_numpy(human_loader.get_human_tracks()).to(device).float()
    human_semantics = torch.from_numpy(human_loader.get_human_semantics()).to(device).float()
    
    print(f"Human tracks shape: {human_tracks.shape}")
    print(f"Human semantics shape: {human_semantics.shape}")
    
    # 初始化robot tracker
    print("Initializing robot tracker...")
    robot_tracker = RobotTracker(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        tapip3d_checkpoint=args.tapip3d_checkpoint,
        siglip_model_path=args.siglip_model_path,
        window_size=args.tracking_window_size,
        step_size=args.tracking_step_size,
        device=device,
        num_targets=args.num_targets,
        image_size=args.siglip_image_size,
        patch_size=args.siglip_patch_size
    )
    
    # 交互式分割robot开始时的物体并提取semantic features
    print("\nPlease segment the objects for robot tracking...")
    first_color, first_depth = agent.get_observation()
    intrinsics_data = agent.intrinsics
    extrinsics_data = np.eye(4)
    
    success = robot_tracker.interactive_segment_objects(
        first_color, first_depth, intrinsics_data, extrinsics_data
    )
    
    if not success:
        print("Failed to segment objects for robot tracking!")
        return
    
    # 获取robot semantic features
    robot_semantics = torch.from_numpy(robot_tracker.get_robot_semantics()).to(device).float()
    print(f"Robot semantics shape: {robot_semantics.shape}")
    
    if args.discretize_rotation:
        last_rot = np.array(agent.ready_rot_6d, dtype=np.float32)
    save_dir = './task_0103_user_0555_scene_0023_cfg_0001_BEFORE_task_0103_user_0555_scene_0025_cfg_0001_AFTER'  # 修改为你的保存路径
    episode_idx = 0  # 你的episode索引
    # 评估循环
    with torch.inference_mode():
        policy.eval()
        prev_width = None
        
        for t in range(1, args.max_steps):
            robot_tcp = agent.robot.get_tcp_pose()
            gripper_width = agent.get_gripper_width()
            
            colors, depths = agent.get_observation()
            
            # 添加帧到robot tracker
            robot_tracker.add_frame(colors, depths, intrinsics_data, extrinsics_data)
            if t % args.num_inference_step == 0:
                coords, feats, cloud = create_input(
                    colors,
                    depths,
                    cam_intrinsics=agent.intrinsics,
                    voxel_size=args.voxel_size
                )
                feats, coords = feats.to(device), coords.to(device)
                cloud_data = ME.SparseTensor(feats, coords)

                # 获取robot tracks (绝对坐标，已normalize)
                extrinsics = torch.tensor(np.array([[ 1., 0., 0.,  0.],
                    [0.,  1., 0., 0.],
                    [0. ,  0.,1.,  0.],
                    [ 0., 0., 0., 1.]])).to(device).float()
                robot_tracks_abs = robot_tracker.get_robot_tracks_for_policy()
                robot_tracks_abs_np = robot_tracks_abs.cpu().numpy()
                extrinsics_np = extrinsics.cpu().numpy()
                robot_tracks_abs_np = human_loader.world_to_camera_coords(robot_tracks_abs_np, extrinsics_np)

                # Convert back to tensor
                # robot_tracks_abs = torch.from_numpy(robot_tracks_abs_np).to(device)

                # Then continue with normalization
                robot_tracks_abs = (robot_tracks_abs_np - TRACK_MIN) / (TRACK_MAX - TRACK_MIN) * 2 - 1
                robot_tracks_abs = torch.from_numpy(robot_tracks_abs).to(device)

                robot_tracks_abs = robot_tracks_abs.unsqueeze(0).to(device).float()  # (1, seq_len, num_points, 3)
                # 计算相对版本 (相对于第一帧)
                if robot_tracks_abs.shape[1] > 0:
                    robot_tracks_unnorm = (robot_tracks_abs.cpu() + 1) / 2 * (TRACK_MAX - TRACK_MIN)  # 先反normalize到绝对坐标
                    first_frame = robot_tracks_unnorm[:, 0:1]  # (1, 1, num_points, 3)
                    robot_tracks_rel_unnorm = robot_tracks_unnorm - first_frame  # 相对运动
                    from dataset.constants import REL_TRACK_MIN, REL_TRACK_MAX
                    robot_tracks_rel = (robot_tracks_rel_unnorm - REL_TRACK_MIN) / (REL_TRACK_MAX - REL_TRACK_MIN) * 2 - 1
                    robot_tracks_rel = torch.clamp(robot_tracks_rel, -1.0, 1.0).to(device).float()
                    print(robot_tracks_rel.shape)
                else:
                    raise ValueError("error")
                
                # Human tracks (已经是normalize后的)
                human_tracks_input_abs = human_tracks.unsqueeze(0).to(device).float()  # (1, seq_len, num_points, 3)
                
                # 计算human的相对版本
                if human_tracks.shape[0] > 0:
                    human_tracks_unnorm = (human_tracks.cpu() + 1) / 2 * (TRACK_MAX - TRACK_MIN)
                    first_frame_human = human_tracks_unnorm[0:1]  # (1, num_points, 3)
                    human_tracks_rel_unnorm = human_tracks_unnorm - first_frame_human
                    human_tracks_rel = (human_tracks_rel_unnorm - REL_TRACK_MIN) / (REL_TRACK_MAX - REL_TRACK_MIN) * 2 - 1
                    human_tracks_rel = torch.clamp(human_tracks_rel, -1.0, 1.0)
                    human_tracks_input_rel = human_tracks_rel.unsqueeze(0).to(device).float()
                    
                
                human_semantics_input = human_semantics.unsqueeze(0)  # (1, num_targets, feature_dim)
                robot_semantics_input = robot_semantics.unsqueeze(0)  # (1, num_targets, feature_dim)


                human_track_lengths = torch.tensor([human_tracks_input_rel.shape[1]], dtype=torch.long, device=device)
                robot_track_lengths = torch.tensor([robot_tracks_rel.shape[1]], dtype=torch.long, device=device)
                robot_total_length_input = torch.tensor([200], dtype=torch.long, device=device)



                # loaded_data = load_tracks_and_semantics(save_dir, episode_idx, device)
                # save_tracks_and_semantics("./val_1", 0,
                #                 human_tracks_input_abs, robot_tracks_abs,
                #               human_tracks_input_rel, robot_tracks_rel,
                #               human_semantics_input, robot_semantics_input,
                #         human_track_lengths, robot_track_lengths,feats,coords)
                
                # human_tracks_input_abs = loaded_data['human_tracks_abs']
                # human_tracks_input_rel = loaded_data['human_tracks_rel']
                # robot_tracks_abs = loaded_data['robot_tracks_abs']
                # robot_tracks_rel = loaded_data['robot_tracks_rel']
                # human_track_lengths = loaded_data['human_track_lengths']
                # robot_track_lengths = loaded_data['robot_track_lengths']
                # human_semantics_input = loaded_data['human_semantics']
                # robot_semantics_input = loaded_data['robot_semantics']
                # robot_total_length_input = loaded_data['robot_total_length']
                # cloud_feats = loaded_data['cloud_feats']
                # cloud_coords = loaded_data['cloud_coords']
                # cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
                # # 保存tracks用于可视化
                # np.savez(
                #     f'./track_results/track_history_{t}.npz',
                #     video=np.expand_dims(colors, axis=0),
                #     depths=np.expand_dims(depths, axis=0),
                #     intrinsics=intrinsics_data,
                #     extrinsics=extrinsics_data,
                #     coords=((robot_tracks_abs.squeeze(0).cpu() + 1) / 2 * (TRACK_MAX - TRACK_MIN) + TRACK_MIN),
                #     visibs=np.ones((robot_tracks_abs.squeeze(0).shape[0], robot_tracks_abs.squeeze(0).shape[1]), dtype=np.float32)
                # )
                
                # Policy推理 - 使用新的参数格式
                pred_raw_action = policy(
                    cloud=cloud_data,
                    actions=None,
                    human_tracks_abs=human_tracks_input_abs,
                    human_tracks_rel=human_tracks_input_rel,
                    robot_tracks_abs=robot_tracks_abs,
                    robot_tracks_rel=robot_tracks_rel,
                    human_track_lengths=human_track_lengths,
                    robot_track_lengths=robot_track_lengths,
                    human_semantics=human_semantics_input,
                    robot_semantics=robot_semantics_input,
                    robot_total_length=robot_total_length_input,
                    batch_size=1
                ).squeeze(0).cpu().numpy()
                
                action = unnormalize_action(pred_raw_action)
                
                # Visualization
                if args.vis:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                    o3d.visualization.draw_geometries([pcd, *tcp_vis_list])
                
                action_width = action[..., -1]
                action_camera = np.concatenate([action[..., :-1], action_width[..., np.newaxis]], axis=-1)
                
                ensemble_buffer.add_action(action_camera, t)
                
            step_action_camera = ensemble_buffer.get_action()
            if step_action_camera is None:
                continue

            step_tcp_camera = step_action_camera[:-1]
            step_width = step_action_camera[-1]
            
            # 投影到base坐标系
            step_tcp_base = projector.project_tcp_to_base_coord(
                step_tcp_camera,
                cam=agent.camera_serial,
                rotation_rep="rotation_6d"
            )
            print(step_tcp_base)
            # input()
            
            # 安全限制
            step_tcp_base[..., :3] = np.clip(
                step_tcp_base[..., :3],
                SAFE_WORKSPACE_MIN + SAFE_EPS,
                SAFE_WORKSPACE_MAX - SAFE_EPS
            )

            # 执行动作
            if args.discretize_rotation:
                rot_steps = discretize_rotation(last_rot, step_tcp_base[3:], np.pi / 16)
                last_rot = step_tcp_base[3:]
                for rot in rot_steps:
                    step_tcp_base[3:] = rot
                    agent.set_tcp_pose(
                        step_tcp_base,
                        rotation_rep="rotation_6d",
                        blocking=True
                    )
            else:
                agent.set_tcp_pose(
                    step_tcp_base,
                    rotation_rep="rotation_6d",
                    blocking=True
                )
            
            if step_width < 0.03:
                step_width = 0
            
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                agent.set_gripper_width(step_width, blocking=True)
                prev_width = step_width
    
    agent.stop()
    print("\nEvaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='checkpoint path', required=True)
    parser.add_argument('--calib', action='store', type=str, help='calibration path', required=True)
    parser.add_argument('--human_demo_path', action='store', type=str, help='path to human demonstration data', required=True)
    parser.add_argument('--siglip_model_path', action='store', type=str, help='path to SigLIP model', required=False, default="/home/ubuntu/data/pretrained-models/siglip-so400m-patch14-384")
    parser.add_argument('--num_action', action='store', type=int, help='number of action steps', required=False, default=20)
    parser.add_argument('--num_history', action='store', type=int, help='number of history relative actions', required=False, default=5)
    parser.add_argument('--num_inference_step', action='store', type=int, help='number of inference query steps', required=False, default=16)
    parser.add_argument('--voxel_size', action='store', type=float, help='voxel size', required=False, default=0.005)
    parser.add_argument('--obs_feature_dim', action='store', type=int, help='observation feature dimension', required=False, default=512)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden dimension', required=False, default=512)
    parser.add_argument('--nheads', action='store', type=int, help='number of heads', required=False, default=8)
    parser.add_argument('--num_encoder_layers', action='store', type=int, help='number of encoder layers', required=False, default=4)
    parser.add_argument('--num_decoder_layers', action='store', type=int, help='number of decoder layers', required=False, default=1)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='feedforward dimension', required=False, default=2048)
    parser.add_argument('--dropout', action='store', type=float, help='dropout ratio', required=False, default=0.1)
    parser.add_argument('--max_steps', action='store', type=int, help='max steps for evaluation', required=False, default=300)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False, default=233)
    parser.add_argument('--vis', action='store_true', help='add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action='store_true', help='whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action='store', type=str, help='temporal ensemble mode', required=False, default='act')
    parser.add_argument('--sam2_checkpoint', action='store', type=str, help='path to SAM2 checkpoint', required=False, default="/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/sam2/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2_config', action='store', type=str, help='path to SAM2 config', required=False, default="//home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/sam2/sam2.1_hiera_l.yaml")
    parser.add_argument('--tapip3d_checkpoint', action='store', type=str, help='path to TAPIP3D checkpoint', required=False, default="/home/ubuntu/git/jingjing/testspace/RISE_add_cotracker_only_object_3d_pos/TAPIP3D/checkpoints/tapip3d_final.pth")
    parser.add_argument('--tracking_window_size', action='store', type=int, help='tracking window size', required=False, default=16)
    parser.add_argument('--tracking_step_size', action='store', type=int, help='tracking step size', required=False, default=8)
    parser.add_argument('--num_targets', action='store', type=int, help='number of targets to use', required=False, default=3)
    parser.add_argument('--points_per_target', action='store', type=int, help='number of points per target', required=False, default=10)
    parser.add_argument('--siglip_image_size', action='store', type=int, help='SigLIP image size', required=False, default=896)
    parser.add_argument('--siglip_patch_size', action='store', type=int, help='SigLIP patch size', required=False, default=14)

    evaluate(vars(parser.parse_args()))