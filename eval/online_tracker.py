import os
import time
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist
import cv2
from PIL import Image
import time
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup
import sys
sys.path.append('TAPIP3D/')

from utils.inference_utils import load_model, inference_with_mask, get_grid_queries, inference
sys.path.append('sam2/')

from sam2.build_sam import build_sam2_video_predictor

from util.constants import (
    IMG_MEAN, IMG_STD, TRANS_MIN, TRANS_MAX, MAX_GRIPPER_WIDTH,
    WORKSPACE_MIN, WORKSPACE_MAX
)

from util.transformation import rotation_transform


class OnlineSAM2TAPIP3DIntegration:
    
    def __init__(self, sam2_checkpoint: str, sam2_config: str, tapip3d_checkpoint: str,
                 window_size: int = 16, step_size: int = 8, device: str = "auto", num_targets: int = 2):
        
        self.device = self._setup_device(device)
        self.window_size = window_size
        self.step_size = step_size
        
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.tapip3d_checkpoint = tapip3d_checkpoint
        
        self.sam2_predictor = None
        self.tapip3d_model = None
        
        self._init_sam2()
        self._init_tapip3d()
        
        from collections import deque
        self.frame_buffer = deque(maxlen=window_size)
        self.depth_buffer = deque(maxlen=window_size)
        self.intrinsics_buffer = deque(maxlen=window_size)
        self.extrinsics_buffer = deque(maxlen=window_size)
        
        self.global_coords = []  
        self.global_visibs = [] 
        self.global_frame_count = 0
        
        self.targets = {}  # target_id -> target_info
        self.current_target_id = None
        self.num_targets = num_targets
        
        self.inference_state = None
        self.temp_dir = None
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        print(f"Using device: {device}")
        return device
    
    def _init_sam2(self):
        self.sam2_predictor = build_sam2_video_predictor(
            self.sam2_config, 
            self.sam2_checkpoint, 
            device=self.device
        )
        
    def _init_tapip3d(self):
        self.tapip3d_model = load_model(self.tapip3d_checkpoint)
        self.tapip3d_model.to(self.device)
        self.tapip3d_model.eval()
        
        if hasattr(self.tapip3d_model, "set_eval_mode"):
            self.tapip3d_model.set_eval_mode("raw")

    def interactive_segment_objects(self, frame: np.ndarray, depth: np.ndarray, 
                                intrinsics: np.ndarray, extrinsics: np.ndarray, 
                                save_dir: str = "./masks") -> bool:
        """Interactive segmentation of two objects and save masks separately"""
        import tempfile
        import shutil
        from pathlib import Path
        from PIL import Image
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        original_img_path = os.path.join(save_dir, "original_frame.png")
        Image.fromarray(frame).save(original_img_path)
        print(f"Original image saved to: {original_img_path}")

        for target_id in range(1, self.num_targets + 1):
            print(f"\n=== Segmenting object {target_id} ===")
            
            temp_dir = tempfile.mkdtemp()
            try:
                frame_path = Path(temp_dir) / "000000.jpg"
                Image.fromarray(frame).save(frame_path)

                # Create independent inference_state for each object
                inference_state = self.sam2_predictor.init_state(video_path=temp_dir)
                
                points = []
                labels = []
                mask = None
                
                plt.figure(figsize=(12, 8))
                plt.title(f"Please select points for object {target_id} (left=positive, right=negative, press q to finish, press z to undo)")
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
                            # Use obj_id=1 (fixed value, since each inference_state only handles one object)
                            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=1,  # Fixed to obj_id=1 in each inference_state
                                points=np.array(points),
                                labels=np.array(labels)
                            )
                            
                            # Clear previous visualization
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
                                
                                # Display selected points
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
                                
                                # Use different colors for different objects
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
                        
                        # Clear current visualization
                        ax = plt.gca()
                        images = ax.images
                        if len(images) > 1:
                            for img in images[1:]:
                                img.remove()
                        for collection in ax.collections:
                            collection.remove()
                        
                        if len(points) > 0:
                            # Re-add remaining points
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
                    print(f"Object {target_id} segmentation failed, please retry")
                    return False
                
                # Save mask
                print(f"Object {target_id} segmentation completed, mask shape: {mask.shape}")
                print(f"Object {target_id} mask True pixel count: {np.sum(mask)}")
                
                # Save binary mask
                mask_filename = f"object_{target_id}_mask.npy"
                mask_path = os.path.join(save_dir, mask_filename)
                np.save(mask_path, mask.astype(np.uint8))
                print(f"Object {target_id} mask saved to: {mask_path}")
                
                # Save visualization mask image
                mask_vis_filename = f"object_{target_id}_mask_vis.png"
                mask_vis_path = os.path.join(save_dir, mask_vis_filename)
                
                # Create colored mask visualization
                mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                color_map = plt.get_cmap("tab10")(target_id)[:3]
                for c in range(3):
                    mask_colored[:, :, c] = mask * (color_map[c] * 255)
                
                Image.fromarray(mask_colored).save(mask_vis_path)
                print(f"Object {target_id} visualization mask saved to: {mask_vis_path}")
                
                # Save original image with mask overlay
                overlay_filename = f"object_{target_id}_overlay.png"
                overlay_path = os.path.join(save_dir, overlay_filename)
                
                # Create overlay image
                overlay = frame.copy().astype(np.float32)
                mask_3d = np.stack([mask] * 3, axis=-1)
                color_overlay = np.array(color_map[:3]) * 255
                overlay = overlay * (1 - mask_3d * 0.4) + mask_3d * color_overlay * 0.4
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                
                Image.fromarray(overlay).save(overlay_path)
                print(f"Object {target_id} overlay image saved to: {overlay_path}")
                
                # Save selected points information
                points_filename = f"object_{target_id}_points.npz"
                points_path = os.path.join(save_dir, points_filename)
                np.savez(points_path, 
                        points=np.array(points), 
                        labels=np.array(labels))
                print(f"Object {target_id} selected points information saved to: {points_path}")
                
                # Generate query points
                query_points = self._generate_query_points_from_mask(
                    torch.from_numpy(mask).to(self.device),
                    torch.from_numpy(depth).to(self.device),
                    torch.from_numpy(intrinsics).to(self.device),
                    torch.from_numpy(extrinsics).to(self.device),
                    num_points=5 
                )
                
                self.targets[target_id] = {
                    'mask': mask,
                    'query_points': query_points,
                    'initialized': True,
                    'last_coords': None,
                    'last_visibs': None
                }
                
                print(f"Object {target_id} segmentation completed, generated {query_points.shape[1]} query points")
                
            finally:
                # Clean up temporary directory for each object
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
        return True
        
    
    def _generate_query_points_from_mask(self, mask: torch.Tensor, depth: torch.Tensor,
                                       intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                                       num_points: int = 20) -> torch.Tensor:
        # Find valid pixels in mask
        intrinsics = intrinsics.float() 
        extrinsics = extrinsics.float()
        mask_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(mask_indices) > num_points:
            sampled_indices = torch.randperm(len(mask_indices))[:num_points]
            mask_indices = mask_indices[sampled_indices]
        
        y_coords = mask_indices[:, 0].float()
        x_coords = mask_indices[:, 1].float()
        
        query_coords_2d = torch.stack([x_coords, y_coords], dim=1)
        
        # Sample depth values
        depths = depth[mask_indices[:, 0], mask_indices[:, 1]]
        
        # Back-project to 3D world coordinates
        inv_intrinsic = torch.linalg.inv(intrinsics)
        inv_extrinsic = torch.linalg.inv(extrinsics)
        
        # Convert to homogeneous coordinates
        query_coords_homo = torch.cat([query_coords_2d, torch.ones(len(query_coords_2d), 1, device=self.device)], dim=1)
        

        camera_coords = torch.einsum('ij,nj->ni', inv_intrinsic, query_coords_homo)
        camera_coords = camera_coords * depths.unsqueeze(1)
        camera_coords_homo = torch.cat([camera_coords, torch.ones(len(camera_coords), 1, device=self.device)], dim=1)
        
        # World coordinate system
        world_coords = torch.einsum('ij,nj->ni', inv_extrinsic, camera_coords_homo)[:, :3]
        
        # Construct query point format: (1, N, 4) - (batch, points, (t, x, y, z))
        query_points = torch.cat([
            torch.zeros(len(world_coords), 1, device=self.device),  # t=0
            world_coords
        ], dim=1).unsqueeze(0)
        
        return query_points
    
    def add_frame(self, frame: np.ndarray, depth: np.ndarray,
                  intrinsics: np.ndarray, extrinsics: np.ndarray) -> dict:
        
        self.frame_buffer.append(frame)
        self.depth_buffer.append(depth)
        self.intrinsics_buffer.append(intrinsics)
        self.extrinsics_buffer.append(extrinsics)
        
        self.global_frame_count += 1
        
        if (self.global_frame_count % self.step_size == 0 and 
            len(self.frame_buffer) >= self.window_size and
            len(self.targets) > 0) or self.global_frame_count==1:
            
            return self._run_inference()
        
        return {}
    
    def _run_inference(self) -> dict:
        # return results
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

            # query_points format: (1, N, 4)
            query_points_squeezed = query_points.squeeze(0)  # (N, 4)
            
            initialized_targets[target_id] = target_info
            all_query_points.append(query_points_squeezed)
            target_point_counts.append(query_points_squeezed.shape[0])
            target_ids_list.append(target_id)
        
        # If no initialized targets, return directly
        if not initialized_targets:
            return results
        
        combined_query_points = torch.cat(all_query_points, dim=0)  # (total_points, 4)
        
        # print(f"Batch inference: {len(initialized_targets)} targets, total {combined_query_points.shape[0]} query points")
        
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
            
            target_coords = coords[:, start_idx:end_idx, :]  # (T, N_target, 3)
            target_visibs = visibs[:, start_idx:end_idx]     # (T, N_target)
            
            target_info = initialized_targets[target_id]
            target_info['last_coords'] = target_coords
            target_info['last_visibs'] = target_visibs
            
            self._update_global_results(target_id, target_coords, target_visibs)
            
            results[target_id] = (
                self._get_global_coords(target_id),
                self._get_global_visibs(target_id)
            )
            
            start_idx = end_idx
            
            # print(f"Target {target_id}: {target_coords.shape[1]} points, shape {target_coords.shape}")
        
        return results
    
    def _get_query_points_for_target(self, target_id: int, current_frame: torch.Tensor) -> torch.Tensor:
        target_info = self.targets[target_id]
        
        if target_info['last_coords'] is not None and target_info['last_visibs'] is not None:
            last_coords = target_info['last_coords'][self.step_size] 
            last_visibs = target_info['last_visibs'][self.step_size]
            
            visible_mask = last_visibs >= 0
            if visible_mask.sum() > 0:
                visible_coords = last_coords[visible_mask]
                
                # Construct query point format: (1, N, 4) - (batch, points, (t, x, y, z))
                query_points = torch.cat([
                    torch.zeros(len(visible_coords), 1, device=self.device),  # t=0
                    visible_coords
                ], dim=1).unsqueeze(0)
                
                return query_points
        
        return target_info['query_points']
    
    def _update_global_results(self, target_id: int, coords: torch.Tensor, visibs: torch.Tensor):
        while len(self.global_coords) <= target_id:
            self.global_coords.append([])
            self.global_visibs.append([])
        
        start_frame = max(0, self.global_frame_count - self.window_size)
        for t in range(coords.shape[0]):
            frame_idx = start_frame + t
            
            if frame_idx < len(self.global_coords[target_id]):
                self.global_coords[target_id][frame_idx] = coords[t]
                self.global_visibs[target_id][frame_idx] = visibs[t]
            else:
                self.global_coords[target_id].append(coords[t])
                self.global_visibs[target_id].append(visibs[t])
    
    def _get_global_coords(self, target_id: int) -> torch.Tensor:
        if target_id >= len(self.global_coords) or len(self.global_coords[target_id]) == 0:
            raise TypeError("error!")

        return torch.stack(self.global_coords[target_id])
    
    def _get_global_visibs(self, target_id: int) -> torch.Tensor:
        if target_id >= len(self.global_visibs) or len(self.global_visibs[target_id]) == 0:
            return torch.empty(0, 0, device=self.device)
        
        return torch.stack(self.global_visibs[target_id])
    
    def get_3d_tracks_for_policy(self) -> torch.Tensor:
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


def create_point_cloud(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    color, depth => point cloud
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
    mask = (x_mask & y_mask & z_mask)
    points = points[mask]
    colors = colors[mask]
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # final cloud
    cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
    return cloud_final

def create_batch(coords, feats):
    """
    coords, feats => batch coords, batch feats (batch size = 1)
    """
    coords_batch = [coords]
    feats_batch = [feats]
    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
    return coords_batch, feats_batch

def create_input(colors, depths, cam_intrinsics, voxel_size = 0.005):
    """
    colors, depths => batch coords, batch feats
    """
    cloud = create_point_cloud(colors, depths, cam_intrinsics, voxel_size = voxel_size)
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype = np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud

def unnormalize_action(action):
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
    return action


def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps