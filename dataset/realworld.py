import os
import json
import torch
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
from typing import List

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import random
from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


def create_variable_length_batch(sequences: List[torch.Tensor], pad_value: float = 0.0):
    batch_size = len(sequences)
    max_seq_len = max(seq.size(0) for seq in sequences)
    num_points = sequences[0].size(1)
    input_dim = sequences[0].size(2)
    
    padded_batch = torch.full(
        (batch_size, max_seq_len, num_points, input_dim),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long, device=sequences[0].device)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_batch[i, :seq_len] = seq
        lengths[i] = seq_len
        
    return padded_batch, lengths


class RealWorldDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        split = 'train', 
        num_obs = 1,
        num_action = 20, 
        num_history = 5, 
        voxel_size = 0.005,
        cam_ids = ['104122063550'],
        aug = False,
        aug_trans_min = [-0.2, -0.2, -0.2],
        aug_trans_max = [0.2, 0.2, 0.2],
        aug_rot_min = [-30, -30, -30],
        aug_rot_max = [30, 30, 30],
        aug_jitter = False,
        aug_jitter_params = [0.4, 0.4, 0.2, 0.1],
        aug_jitter_prob = 0.2,
        with_cloud = False,
        vis = False,
        num_targets = 2,
        points_per_target = 10
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.num_history = num_history
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min)
        self.aug_trans_max = np.array(aug_trans_max)
        self.aug_rot_min = np.array(aug_rot_min)
        self.aug_rot_max = np.array(aug_rot_max)
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params)
        self.aug_jitter_prob = aug_jitter_prob
        self.with_cloud = with_cloud
        self.vis = vis
        self.num_targets = num_targets
        self.points_per_target = points_per_target
        
        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.cam_ids = []
        self.calib_timestamp = []
        self.obs_frame_ids = []
        self.action_frame_ids = []
        self.history_frame_ids = []
        self.track_indices = []
        self.demo_robot_start_indices = {}
        
        # 修改缓存策略：分别缓存human和robot的tracks和semantics
        self.human_track_cache = {}  # key: (scene_name, cam_id)
        self.robot_track_cache = {}  # key: (scene_name, cam_id)
        self.projectors = {}
        
        # 预加载所有demo的robot_start_idx
        print("Preloading robot_start_idx for all demos...")
        for demo_name in tqdm(self.all_demos):
            demo_path = os.path.join(self.data_path, demo_name)
            human_json_path = os.path.join(demo_path, "human.json")
            
            if os.path.exists(human_json_path):
                with open(human_json_path, "r") as f:
                    human_data = json.load(f)
                    robot_start_idx = int(human_data.get("robot_start_idx", 0))
            else:
                raise FileNotFoundError(f"human.json not found in {demo_path}")
            
            # 存储每个demo的robot_start_idx
            self.demo_robot_start_indices[demo_path] = robot_start_idx
        
        # 构建数据集样本
        print("Building dataset samples...")
        for i in tqdm(range(self.num_demos)):
            demo_name = self.all_demos[i]
            demo_path = os.path.join(self.data_path, demo_name)
            robot_start_idx = self.demo_robot_start_indices[demo_path]
            human_json_path = os.path.join(demo_path, "human.json")
            robot_start_time = None
            if os.path.exists(human_json_path):
                with open(human_json_path, "r") as f:
                    human_data = json.load(f)
                    robot_start_time = int(human_data.get("robor_start_time", "0"))
            else:
                print(f"Warning: human.json not found at {human_json_path}")
                robot_start_time = None
            for cam_id in cam_ids:
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                    
                with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                    meta = json.load(f)
                    
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                ]
                
                with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                    calib_timestamp = f.readline().rstrip()
                
                obs_frame_ids_list = []
                action_frame_ids_list = []
                history_frame_ids_list = []
                track_indices_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    if cur_idx < robot_start_idx:
                        continue
                    current_timestamp = frame_ids[cur_idx]
                    if current_timestamp <= robot_start_time:
                        continue  # 跳过这个cur_idx，不进入后续处理流程

                    obs_pad_before = max(0, num_obs - cur_idx - 1)
                    action_pad_after = max(0, num_action - (len(frame_ids) - 1 - cur_idx))
                    frame_begin = max(0, cur_idx - num_obs + 1)
                    frame_end = min(len(frame_ids), cur_idx + num_action + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    action_frame_ids = frame_ids[cur_idx + 1: frame_end] + frame_ids[-1:] * action_pad_after
                    
                    history_begin = max(0, cur_idx - self.num_history)
                    actual_history_frames = frame_ids[history_begin:cur_idx + 1]  
                    
                    if len(actual_history_frames) < self.num_history + 1:
                        padding_count = self.num_history + 1 - len(actual_history_frames)
                        history_frame_ids = [frame_ids[0]] * padding_count + actual_history_frames
                    else:
                        history_frame_ids = actual_history_frames
                    
                    assert len(history_frame_ids) == self.num_history + 1, \
                        f"Expected {self.num_history + 1} history frames, got {len(history_frame_ids)}"
                    
                    obs_frame_ids_list.append(obs_frame_ids)
                    action_frame_ids_list.append(action_frame_ids)
                    history_frame_ids_list.append(history_frame_ids)
                    track_indices_list.append(cur_idx)

                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.calib_timestamp += [calib_timestamp] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.action_frame_ids += action_frame_ids_list
                self.history_frame_ids += history_frame_ids_list
                self.track_indices += track_indices_list
        
        print(f"Dataset initialized with {len(self.data_paths)} samples")
        
    def __len__(self):
        return len(self.obs_frame_ids)


    def _augmentation(self, clouds, tcps, human_tracks_abs=None, human_tracks_rel=None, 
                    robot_tracks_abs=None, robot_tracks_rel=None):
        """
        对点云、TCP和tracks进行一致的数据增强
        
        Args:
            clouds: list of point clouds with shape (N, 6) - xyz + rgb
            tcps: action TCPs with shape (T, 7) - xyz + quaternion
            human_tracks_abs: (T_human, num_targets*num_points, 3) - absolute normalized in [-1, 1]
            human_tracks_rel: (T_human, num_targets*num_points, 3) - relative normalized in [-1, 1]
            robot_tracks_abs: (T_robot, num_targets*num_points, 3) - absolute normalized in [-1, 1]
            robot_tracks_rel: (T_robot, num_targets*num_points, 3) - relative normalized in [-1, 1]
        
        Returns:
            augmented clouds, tcps, human_tracks_abs, human_tracks_rel, robot_tracks_abs, robot_tracks_rel
        """
        # 生成随机变换参数
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # transform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        
        # 使用最后一帧点云的中心作为旋转中心
        center = clouds[-1][..., :3].mean(axis=0)

        # 增强点云
        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        # 增强TCPs
        tcps[..., :3] -= center
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep="quaternion")
        tcps[..., :3] += center

        # 增强human tracks (绝对坐标)
        if human_tracks_abs is not None:
            human_tracks_abs = self._augment_tracks(human_tracks_abs, aug_mat, center)
        
        # 增强human tracks (相对坐标)
        if human_tracks_rel is not None:
            human_tracks_rel = self._augment_relative_tracks(human_tracks_rel, aug_mat)
        
        # 增强robot tracks (绝对坐标)
        if robot_tracks_abs is not None:
            robot_tracks_abs = self._augment_tracks(robot_tracks_abs, aug_mat, center)
        
        # 增强robot tracks (相对坐标)
        if robot_tracks_rel is not None:
            robot_tracks_rel = self._augment_relative_tracks(robot_tracks_rel, aug_mat)

        return clouds, tcps, human_tracks_abs, human_tracks_rel, robot_tracks_abs, robot_tracks_rel

    def _augment_tracks(self, tracks, aug_mat, center):
        """
        对tracks应用数据增强变换（与点云使用相同的方法）
        
        Args:
            tracks: (T, N, 3) - normalized tracks in [-1, 1]
            aug_mat: (4, 4) - augmentation transformation matrix
            center: (3,) - rotation center
        
        Returns:
            augmented_tracks: (T, N, 3) - augmented normalized tracks
        """
        # 1. Denormalize到绝对坐标（米）
        tracks_abs = (tracks + 1) / 2 * (TRACK_MAX - TRACK_MIN) + TRACK_MIN
        
        # 2. 保存原始shape
        original_shape = tracks_abs.shape  # (T, N, 3)
        
        # 3. Reshape成2D: (T*N, 3) 以便使用 apply_mat_to_pcd
        tracks_flat = tracks_abs.reshape(-1, 3)  # (T*N, 3)
        
        # 4. 去中心化
        tracks_flat[..., :3] -= center
        
        # 5. 使用与点云相同的函数应用变换
        tracks_flat = apply_mat_to_pcd(tracks_flat, aug_mat)
        
        # 6. 加回中心
        tracks_flat[..., :3] += center
        
        # 7. Reshape回原始shape
        tracks_abs = tracks_flat.reshape(original_shape)
        
        # 8. Normalize回[-1, 1]
        tracks_normalized = (tracks_abs - TRACK_MIN) / (TRACK_MAX - TRACK_MIN) * 2 - 1
        tracks_normalized = np.clip(tracks_normalized, -1.0, 1.0)
        
        return tracks_normalized

    def _augment_relative_tracks(self, tracks_rel, aug_mat):
        """
        对相对坐标tracks应用数据增强（只应用旋转，不应用平移）
        
        Args:
            tracks_rel: (T, N, 3) - normalized relative tracks in [-1, 1]
            aug_mat: (4, 4) - augmentation transformation matrix
        
        Returns:
            augmented_tracks_rel: (T, N, 3) - augmented normalized relative tracks
        """
        from utils.constants import REL_TRACK_MIN, REL_TRACK_MAX
        tracks_rel_denorm = (tracks_rel + 1) / 2 * (REL_TRACK_MAX - REL_TRACK_MIN) + REL_TRACK_MIN
        
        original_shape = tracks_rel_denorm.shape  # (T, N, 3)
        
        # (T*N, 3)
        tracks_flat = tracks_rel_denorm.reshape(-1, 3)  # (T*N, 3)
        
        rot_mat = aug_mat[:3, :3]
        tracks_flat = (rot_mat @ tracks_flat.T).T

        tracks_rel_denorm = tracks_flat.reshape(original_shape)
        
        tracks_rel_normalized = (tracks_rel_denorm - REL_TRACK_MIN) / (REL_TRACK_MAX - REL_TRACK_MIN) * 2 - 1
        tracks_rel_normalized = np.clip(tracks_rel_normalized, -1.0, 1.0)
        
        return tracks_rel_normalized

    def _normalize_tcp(self, tcp_list):
        """Normalize TCP: [T, 3(trans) + 6(rot) + 1(width)]"""
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
        return tcp_list

    def load_point_cloud(self, colors, depths, cam_id):
        """Load and voxelize point cloud from RGB-D"""
        h, w = depths.shape
        fx, fy = INTRINSICS[cam_id][0, 0], INTRINSICS[cam_id][1, 1]
        cx, cy = INTRINSICS[cam_id][0, 2], INTRINSICS[cam_id][1, 2]
        scale = 1000. if 'f' not in cam_id else 4000.
        
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, scale, convert_rgb_to_intensity=False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        points = np.array(cloud.points)
        colors = np.array(cloud.colors)
        return points.astype(np.float32), colors.astype(np.float32)

    def fps_sampling_3d(self, points, n_samples):
        """3D最远点采样"""
        N = points.shape[0]
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(N) * 1e10
        
        # farthest = np.random.randint(0, N)
        farthest = 0
        
        for i in range(n_samples):
            sampled_indices[i] = farthest
            centroid = points[farthest]
            dist = np.sum((points - centroid) ** 2, axis=1)
            
            distances = np.minimum(distances, dist)
            
            # 选择距离最远的点
            farthest = np.argmax(distances)
        
        return sampled_indices

    def _extract_scene_names(self, demo_name):
        """
        (before_scene, after_scene)
        """
        parts = demo_name.split('_BEFORE_')
        if len(parts) != 2:
            raise ValueError(f"Invalid demo name format: {demo_name}")
        
        before_scene = parts[0]
        after_scene = parts[1].replace('_AFTER', '')
        
        return before_scene, after_scene
            

    def world_to_camera_coords(self, world_coords, extrinsics):
        original_shape = world_coords.shape
        
        # Flatten to (M, 3) where M = T*N or N
        if world_coords.ndim == 3:
            T, N, _ = world_coords.shape
            world_coords_flat = world_coords.reshape(-1, 3)
        else:
            world_coords_flat = world_coords
        
        # (M, 4)
        ones = np.ones((world_coords_flat.shape[0], 1), dtype=world_coords_flat.dtype)
        world_coords_homo = np.concatenate([world_coords_flat, ones], axis=1)
        
        # camera = inv(extrinsics) @ world
        inv_extrinsics = np.linalg.inv(extrinsics)
        camera_coords_homo = (inv_extrinsics @ world_coords_homo.T).T
        camera_coords = camera_coords_homo[:, :3]
        
        # Reshape
        if len(original_shape) == 3:
            camera_coords = camera_coords.reshape(original_shape)
        
        return camera_coords

    # In RealWorldDataset class, modify _load_human_tracks method:

    def _load_human_tracks(self, demo_path, cam_id):
        """
        Load human tracks - returns both absolute and relative versions
        Returns:
            absolute: normalized absolute coordinates [-1, 1]
            relative: normalized relative motion from first frame [-1, 1]
        """
        demo_name = os.path.basename(demo_path)
        before_scene, _ = self._extract_scene_names(demo_name)
        cache_key = (before_scene, cam_id, self.num_targets)
        
        if cache_key in self.human_track_cache:
            return self.human_track_cache[cache_key]
        
        cam_path = os.path.join(demo_path, f"cam_{cam_id}")
        human_tracks_dir = os.path.join(cam_path, "before_sam2_tapip3d_results_offline")
        human_semantic_dir = os.path.join(cam_path, "human_siglip")
        
        # Load extrinsics
        with open(os.path.join(demo_path, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        color_dir = os.path.join(cam_path, "color")
        frame_ids = sorted([int(os.path.splitext(x)[0]) for x in os.listdir(color_dir) 
                        if int(os.path.splitext(x)[0]) <= meta["finish_time"]])
        first_frame_id = frame_ids[0]
        
        extrinsics = np.array([[ 1., 0., 0.,  0.],
                            [0.,  1., 0., 0.],
                            [0. ,  0.,1.,  0.],
                            [ 0., 0., 0., 1.]])
        
        human_point_tracks_abs = []
        human_point_tracks_rel = []
        human_semantic_features = []
        
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
            human_point_indices = self.fps_sampling_3d(human_first_frame_points, self.points_per_target)
            human_tracks = human_pred_tracks[:, human_point_indices, :]  # (T, num_points, 3)
            
            # === Compute absolute version ===
            human_tracks_abs_normalized = (human_tracks - TRACK_MIN) / (TRACK_MAX - TRACK_MIN) * 2 - 1
            human_tracks_abs_normalized = np.clip(human_tracks_abs_normalized, -1.0, 1.0)
            
            # === Compute relative version ===
            first_frame = human_tracks[0:1]  # (1, num_points, 3)
            human_tracks_relative = human_tracks - first_frame  # (T, num_points, 3) - relative motion
            human_tracks_rel_normalized = (human_tracks_relative - REL_TRACK_MIN) / (REL_TRACK_MAX - REL_TRACK_MIN) * 2 - 1
            human_tracks_rel_normalized = np.clip(human_tracks_rel_normalized, -1.0, 1.0)
            
            # Load semantics
            human_semantic_path = os.path.join(human_semantic_dir, f"target_{target_idx}.npy")
            if not os.path.exists(human_semantic_path):
                raise FileNotFoundError(f"Human semantic file not found")
            human_semantic = np.load(human_semantic_path)
            
            human_point_tracks_abs.append(human_tracks_abs_normalized)
            human_point_tracks_rel.append(human_tracks_rel_normalized)
            human_semantic_features.append(human_semantic)
        
        final_human_tracks_abs = np.concatenate(human_point_tracks_abs, axis=1)
        final_human_tracks_rel = np.concatenate(human_point_tracks_rel, axis=1)
        final_human_semantics = np.stack(human_semantic_features, axis=1)
        
        # Cache both versions
        self.human_track_cache[cache_key] = (final_human_tracks_abs, final_human_tracks_rel, final_human_semantics)
        
        return final_human_tracks_abs, final_human_tracks_rel, final_human_semantics


    def _load_robot_tracks(self, demo_path, cam_id):
        """
        Load robot tracks 
        """
        demo_name = os.path.basename(demo_path)
        _, after_scene = self._extract_scene_names(demo_name)
        cache_key = (after_scene, cam_id, self.num_targets)
        
        # === 检查缓存 ===
        if cache_key in self.robot_track_cache:
            return self.robot_track_cache[cache_key]
        
        cam_path = os.path.join(demo_path, f"cam_{cam_id}")
        robot_tracks_dir = os.path.join(cam_path, "after_sam2_tapip3d_results_offline")
        robot_semantic_dir = os.path.join(cam_path, "robot_siglip")
        
        # Load extrinsics
        extrinsics = np.array([[ 1., 0., 0.,  0.],
                            [0.,  1., 0., 0.],
                            [0. ,  0.,1.,  0.],
                            [ 0., 0., 0., 1.]])
        
        robot_point_tracks_abs = []
        robot_point_tracks_rel = []
        robot_semantic_features = []
        
        for target_idx in range(1, self.num_targets + 1):
            # Load tracks
            robot_target_path = os.path.join(robot_tracks_dir, f"3d_tracks_target_after_{target_idx}.npy")

            robot_pred_tracks = np.load(robot_target_path)
            if robot_pred_tracks.ndim == 4:
                robot_pred_tracks = robot_pred_tracks[0]
            
            if robot_pred_tracks.shape[1] < self.points_per_target:
                raise ValueError(f"Not enough points in robot target_{target_idx}")
            
            # Convert to camera coords
            robot_pred_tracks = self.world_to_camera_coords(robot_pred_tracks, extrinsics)
            
            # FPS sampling
            robot_first_frame_points = robot_pred_tracks[0]
            robot_point_indices = self.fps_sampling_3d(robot_first_frame_points, self.points_per_target)
            robot_tracks = robot_pred_tracks[:, robot_point_indices, :]
            
            # === Compute absolute version ===
            robot_tracks_abs_normalized = (robot_tracks - TRACK_MIN) / (TRACK_MAX - TRACK_MIN) * 2 - 1
            robot_tracks_abs_normalized = np.clip(robot_tracks_abs_normalized, -1.0, 1.0)
            
            # === Compute relative version ===
            first_frame = robot_tracks[0:1]
            robot_tracks_relative = robot_tracks - first_frame
            robot_tracks_rel_normalized = (robot_tracks_relative - REL_TRACK_MIN) / (REL_TRACK_MAX - REL_TRACK_MIN) * 2 - 1
            robot_tracks_rel_normalized = np.clip(robot_tracks_rel_normalized, -1.0, 1.0)
            
            # Load semantics
            robot_semantic_path = os.path.join(robot_semantic_dir, f"target_{target_idx}.npy")
            robot_semantic = np.load(robot_semantic_path)
            
            robot_point_tracks_abs.append(robot_tracks_abs_normalized)
            robot_point_tracks_rel.append(robot_tracks_rel_normalized)
            robot_semantic_features.append(robot_semantic)
        
        final_robot_tracks_abs = np.concatenate(robot_point_tracks_abs, axis=1)
        final_robot_tracks_rel = np.concatenate(robot_point_tracks_rel, axis=1)
        final_robot_semantics = np.stack(robot_semantic_features, axis=0)  # (num_targets, D)
        
        # Cache
        self.robot_track_cache[cache_key] = (final_robot_tracks_abs, final_robot_tracks_rel, final_robot_semantics)
        
        return final_robot_tracks_abs, final_robot_tracks_rel, final_robot_semantics


    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        calib_timestamp = self.calib_timestamp[index]
        obs_frame_ids = self.obs_frame_ids[index]
        action_frame_ids = self.action_frame_ids[index]
        history_frame_ids = self.history_frame_ids[index]
        track_idx = self.track_indices[index]
        
        robot_start_idx = self.demo_robot_start_indices[data_path]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')
        tcp_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'tcp')
        gripper_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'gripper_command')

        # load camera projector
        timestamp_path = os.path.join(data_path, 'timestamp.txt')
        with open(timestamp_path, 'r') as f:
            timestamp = f.readline().rstrip()
        if timestamp not in self.projectors:
            self.projectors[timestamp] = Projector(os.path.join(self.calib_path, timestamp))
        projector = self.projectors[timestamp]

        # create color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter = T.ColorJitter(
                brightness = self.aug_jitter_params[0],
                contrast = self.aug_jitter_params[1],
                saturation = self.aug_jitter_params[2],
                hue = self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_jitter_prob)

        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_jitter:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            points, colors = self.load_point_cloud(colors_list[i], depths_list[i], cam_id)
            x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
            # apply imagenet normalization
            colors = (colors - IMG_MEAN) / IMG_STD
            cloud = np.concatenate([points, colors], axis = -1)
            clouds.append(cloud)

        human_tracks_abs, human_tracks_rel, human_semantics = self._load_human_tracks(data_path, cam_id)
        robot_tracks_abs, robot_tracks_rel, robot_semantics = self._load_robot_tracks(data_path, cam_id)
        
        robot_full_length = robot_tracks_abs.shape[0]
        
        # end_idx = max(1, track_idx + 1 - robot_start_idx - random.randint(0, 15))
        end_idx = max(1, track_idx + 1 - robot_start_idx)
        
        full_human_tracks_abs = human_tracks_abs
        full_human_tracks_rel = human_tracks_rel

        if end_idx > robot_tracks_abs.shape[0]:
            raise TypeError("error")
        full_robot_tracks_abs = robot_tracks_abs[:end_idx]
        full_robot_tracks_rel = robot_tracks_rel[:end_idx]


        # actions
        action_tcps = []
        action_grippers = []
        for frame_id in action_frame_ids:
            tcp = np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[:7].astype(np.float32)
            projected_tcp = projector.project_tcp_to_camera_coord(tcp, cam_id)
            gripper_width = decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[0])
            action_tcps.append(projected_tcp)
            action_grippers.append(gripper_width)
        action_tcps = np.stack(action_tcps)
        action_grippers = np.stack(action_grippers)

        # point augmentations
        if self.split == 'train' and self.aug:
            clouds, action_tcps, full_human_tracks_abs, full_human_tracks_rel, full_robot_tracks_abs, full_robot_tracks_rel = self._augmentation(
                clouds, 
                action_tcps,
                human_tracks_abs=full_human_tracks_abs,
                human_tracks_rel=full_human_tracks_rel,
                robot_tracks_abs=full_robot_tracks_abs,
                robot_tracks_rel=full_robot_tracks_rel
            )
        # visualization
        if self.vis:
            points = clouds[-1][..., :3]
            print("point range", points.min(axis=0), points.max(axis=0))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors * IMG_STD + IMG_MEAN)
            
            traj = []
            bbox3d_1 = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
            bbox3d_1.color = [1, 0, 0]
            bbox3d_2 = o3d.geometry.AxisAlignedBoundingBox(TRANS_MIN, TRANS_MAX)
            bbox3d_2.color = [0, 1, 0]
            bbox3d_3 = o3d.geometry.AxisAlignedBoundingBox(TRACK_MIN, TRACK_MAX)
            bbox3d_3.color = [0, 0, 1]
            
            action_tcps_vis = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "matrix")
            for i in range(len(action_tcps_vis)):
                action = action_tcps_vis[i]
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(action)
                traj.append(frame)
            
            self.num_points = 10
            vis_geometries = [pcd.voxel_down_sample(self.voxel_size), bbox3d_1, bbox3d_2, bbox3d_3, *traj]
            
            def denormalize_tracks_absolute(normalized_tracks):
                return (normalized_tracks + 1) / 2 * (TRACK_MAX - TRACK_MIN) + TRACK_MIN
            
            human_tracks_denorm = denormalize_tracks_absolute(full_human_tracks_abs)  # (T, num_targets*num_points, 3)
            T_human = human_tracks_denorm.shape[0]
            human_tracks_reshaped = human_tracks_denorm.reshape(T_human, self.num_targets, self.num_points, 3)
            
            print(f"\n=== Human Tracks Stats ===")
            print(f"Shape: {human_tracks_reshaped.shape}")
            print(f"Range X: [{human_tracks_reshaped[..., 0].min():.3f}, {human_tracks_reshaped[..., 0].max():.3f}]")
            print(f"Range Y: [{human_tracks_reshaped[..., 1].min():.3f}, {human_tracks_reshaped[..., 1].max():.3f}]")
            print(f"Range Z: [{human_tracks_reshaped[..., 2].min():.3f}, {human_tracks_reshaped[..., 2].max():.3f}]")
            
            import matplotlib.pyplot as plt
            cmap = plt.cm.rainbow
            
            for target_idx in range(self.num_targets):
                color = cmap(target_idx / max(self.num_targets, 2))[:3] 
                
                for point_idx in range(self.num_points):
                    point_traj = human_tracks_reshaped[:, target_idx, point_idx, :]
                    
                    if T_human < 2:
                        continue
                    
                    lines = [[i, i+1] for i in range(T_human-1)]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(point_traj)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_colors = [color for _ in range(len(lines))]
                    line_set.colors = o3d.utility.Vector3dVector(line_colors)
                    vis_geometries.append(line_set)
                    
                    sphere_start = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    sphere_start.translate(point_traj[0])
                    sphere_start.paint_uniform_color(color)
                    vis_geometries.append(sphere_start)
                    
                    sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                    sphere_end.translate(point_traj[-1])
                    sphere_end.paint_uniform_color([c * 0.6 for c in color]) 
                    vis_geometries.append(sphere_end)
            

            robot_tracks_denorm = denormalize_tracks_absolute(full_robot_tracks_abs)  # (T_robot, num_targets*num_points, 3)
            T_robot = robot_tracks_denorm.shape[0]
            robot_tracks_reshaped = robot_tracks_denorm.reshape(T_robot, self.num_targets, self.num_points, 3)
            
            print(f"\n=== Robot Tracks Stats ===")
            print(f"Shape: {robot_tracks_reshaped.shape}")
            print(f"Range X: [{robot_tracks_reshaped[..., 0].min():.3f}, {robot_tracks_reshaped[..., 0].max():.3f}]")
            print(f"Range Y: [{robot_tracks_reshaped[..., 1].min():.3f}, {robot_tracks_reshaped[..., 1].max():.3f}]")
            print(f"Range Z: [{robot_tracks_reshaped[..., 2].min():.3f}, {robot_tracks_reshaped[..., 2].max():.3f}]")
            
            for target_idx in range(self.num_targets):
                color = cmap((target_idx + 0.5) / max(self.num_targets, 2))[:3]
                color = (color[0] * 0.5, color[1] * 0.5, color[2] * 1.0) 
                
                for point_idx in range(self.num_points):
                    point_traj = robot_tracks_reshaped[:, target_idx, point_idx, :]
                    
                    if T_robot < 2:
                        continue
                    
                    lines = [[i, i+1] for i in range(T_robot-1)]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(point_traj)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_colors = [color for _ in range(len(lines))]
                    line_set.colors = o3d.utility.Vector3dVector(line_colors)
                    vis_geometries.append(line_set)
                    
                    cube_start = o3d.geometry.TriangleMesh.create_box(width=0.015, height=0.015, depth=0.015)
                    cube_start.translate(point_traj[0] - [0.0075, 0.0075, 0.0075])
                    cube_start.paint_uniform_color(color)
                    vis_geometries.append(cube_start)
                    
                    cube_end = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.02, depth=0.02)
                    cube_end.translate(point_traj[-1] - [0.01, 0.01, 0.01])
                    cube_end.paint_uniform_color([c * 0.6 for c in color])
                    vis_geometries.append(cube_end)
            
            print(f"\n=== Visualization Info ===")
            print(f"Human tracks: {T_human} frames, {self.num_targets} targets, {self.num_points} points per target")
            print(f"Robot tracks: {T_robot} frames, {self.num_targets} targets, {self.num_points} points per target")
            print(f"Human tracks are shown with spheres (warm colors)")
            print(f"Robot tracks are shown with cubes (cool colors)")
            print(f"Red bbox: WORKSPACE range")
            print(f"Green bbox: TRANS range")
            print(f"Blue bbox: TRACK range")
            print("==========================\n")
            
            o3d.visualization.draw_geometries(vis_geometries)
        # rotation transformation
        action_tcps = xyz_rot_transform(action_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        actions = np.concatenate((action_tcps, action_grippers[..., np.newaxis]), axis = -1)
        actions_normalized = self._normalize_tcp(actions.copy())

        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        # Convert tracks to tensors
        human_tracks_abs_tensor = torch.from_numpy(full_human_tracks_abs).float()
        human_tracks_rel_tensor = torch.from_numpy(full_human_tracks_rel).float()
        robot_tracks_abs_tensor = torch.from_numpy(full_robot_tracks_abs).float()
        robot_tracks_rel_tensor = torch.from_numpy(full_robot_tracks_rel).float()
        # print(human_semantics.shape)
        # print(robot_semantics.shape)
        human_semantics_tensor = torch.from_numpy(human_semantics[0,:,:]).float()
        # robot_semantics_tensor = torch.from_numpy(robot_semantics[:, end_idx-1, :]).float()
        robot_semantics_tensor = torch.from_numpy(robot_semantics[:, 0, :]).float()

        
        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': actions,
            'action_normalized': actions_normalized,
            'human_tracks_abs': human_tracks_abs_tensor,
            'human_tracks_rel': human_tracks_rel_tensor,
            'robot_tracks_abs': robot_tracks_abs_tensor,
            'robot_tracks_rel': robot_tracks_rel_tensor,
            'human_semantics': human_semantics_tensor,
            'robot_semantics': robot_semantics_tensor,
            'robot_start_idx': robot_start_idx,
            'robot_total_length': robot_full_length
        }
        
        if self.with_cloud:
            ret_dict["clouds_list"] = clouds

        return ret_dict
            

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                if key == 'human_semantics':
                    human_semantics_list = [d[key] for d in batch]
                    lengths = [hs.size(0) for hs in human_semantics_list]
                    if len(set(lengths)) == 1:
                        ret_dict[key] = torch.stack(human_semantics_list, 0)
                    else:
                        padded_batch, _ = create_variable_length_batch(human_semantics_list)
                        ret_dict[key] = padded_batch
                
                elif key == 'human_tracks_abs' or key == 'human_tracks_rel':
                    human_tracks_list = [d[key] for d in batch]
                    lengths = [ht.size(0) for ht in human_tracks_list]
                    if len(set(lengths)) == 1:
                        ret_dict[key] = torch.stack(human_tracks_list, 0)
                        ret_dict['human_track_lengths'] = torch.full((len(batch),), human_tracks_list[0].shape[0], dtype=torch.long)
                    else:
                        padded_batch, track_lengths = create_variable_length_batch(human_tracks_list)
                        ret_dict[key] = padded_batch
                        ret_dict['human_track_lengths'] = track_lengths
                        
                elif key == 'robot_tracks_abs' or key == 'robot_tracks_rel':
                    robot_tracks_list = [d[key] for d in batch]
                    lengths = [rt.size(0) for rt in robot_tracks_list]
                    if len(set(lengths)) == 1:
                        ret_dict[key] = torch.stack(robot_tracks_list, 0)
                        ret_dict['robot_track_lengths'] = torch.full((len(batch),), robot_tracks_list[0].shape[0], dtype=torch.long)
                    else:
                        padded_batch, track_lengths = create_variable_length_batch(robot_tracks_list)
                        ret_dict[key] = padded_batch
                        ret_dict['robot_track_lengths'] = track_lengths
                else:
                    ret_dict[key] = collate_fn([d[key] for d in batch])
            elif key == 'robot_total_length':
                ret_dict[key] = torch.tensor([d[key] for d in batch], dtype=torch.long)
            else:
                ret_dict[key] = [d[key] for d in batch]
                
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095