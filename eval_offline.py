# eval_offline.py

import os
import json
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from visual_attention import AttentionVisualizer

from policy import HistRISE
from dataset.realworld import RealWorldDataset, collate_fn
from utils.constants import (
    IMG_MEAN, IMG_STD, TRANS_MIN, TRANS_MAX, MAX_GRIPPER_WIDTH,
    WORKSPACE_MIN, WORKSPACE_MAX
)
from utils.training import set_seed
from utils.transformation import rotation_transform


default_args = edict({
    "data_path": "data/collect_pens",
    "ckpt": None,
    "num_action": 20,
    "num_history": 5,
    "num_inference_step": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "seed": 233,
    "max_test_steps": 50,
    "scene_filter": None,
    "vis": False,
    "save_results": True,
    "output_dir": "./eval_results",
    "num_targets": 4,
    "num_points": 10,
})


import torch
import os

# ============= 存储代码 =============
def save_tracks_and_semantics(save_dir, episode_idx, 
                                human_tracks_abs, robot_tracks_abs,
                              human_tracks_rel, robot_tracks_rel,
                              human_semantics, robot_semantics,
                              human_track_lengths=None, 
                              robot_track_lengths=None,robot_total_length=None,cloud_feats=None,cloud_coords=None):
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
        'human_track_lengths': human_track_lengths.cpu() if human_track_lengths is not None else None,
        'robot_track_lengths': robot_track_lengths.cpu() if robot_track_lengths is not None else None,
        'robot_total_length': robot_total_length.cpu() if robot_total_length is not None else None,
        # 添加cloud_feats和cloud_coords
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

class OfflineEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        set_seed(args.seed)
        
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
        
        self.init_model()
        self.init_dataset()
        
        self.results = {
            'predictions': [],
            'ground_truth': [],
            'losses': [],
            'step_info': []
        }
        
    def init_model(self):
        """初始化模型"""
        print("Loading policy...")
        
        # 与train.py保持完全一致的track_config
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

        self.policy = HistRISE(
            num_action=self.args.num_action,
            num_history=self.args.num_history,
            input_dim=6,
            obs_feature_dim=self.args.obs_feature_dim,
            action_dim=10,
            hidden_dim=self.args.hidden_dim,
            nheads=self.args.nheads,
            num_encoder_layers=self.args.num_encoder_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            dropout=self.args.dropout,
            track_config=track_config,
            num_targets=self.args.num_targets,
            num_points=self.args.num_points
        ).to(self.device)
        

        n_parameters = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
        
        if self.args.ckpt is not None:
            checkpoint = torch.load(self.args.ckpt, map_location=self.device)
            self.policy.load_state_dict(checkpoint, strict=False)
            print("Checkpoint {} loaded.".format(self.args.ckpt))
        else:
            print("Warning: No checkpoint provided, using random weights!")
        
        self.policy.eval()
    
    def init_dataset(self):
        """初始化数据集"""
        print("Loading dataset...")
        
        self.dataset = RealWorldDataset(
            path=self.args.data_path,
            split='train',
            num_obs=1,
            num_action=self.args.num_action,
            num_history=self.args.num_history,
            voxel_size=self.args.voxel_size,
            aug=False,
            aug_jitter=False,
            with_cloud=True,
            vis=False,
            num_targets=self.args.num_targets,
            points_per_target=self.args.num_points
        )
        
        # 过滤数据集
        if self.args.scene_filter:
            filtered_indices = []
            for i, data_path in enumerate(self.dataset.data_paths):
                if self.args.scene_filter in data_path:
                    filtered_indices.append(i)
            
            filtered_indices = filtered_indices[:self.args.max_test_steps]
            
            self.dataset.data_paths = [self.dataset.data_paths[i] for i in filtered_indices]
            self.dataset.cam_ids = [self.dataset.cam_ids[i] for i in filtered_indices]
            self.dataset.calib_timestamp = [self.dataset.calib_timestamp[i] for i in filtered_indices]
            self.dataset.obs_frame_ids = [self.dataset.obs_frame_ids[i] for i in filtered_indices]
            self.dataset.action_frame_ids = [self.dataset.action_frame_ids[i] for i in filtered_indices]
            self.dataset.history_frame_ids = [self.dataset.history_frame_ids[i] for i in filtered_indices]
            self.dataset.track_indices = [self.dataset.track_indices[i] for i in filtered_indices]
            
            print(f"Filtered dataset to {len(self.dataset.data_paths)} samples from {self.args.scene_filter}")
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        print(f"Test dataset loaded with {len(self.dataset)} samples")
    
    def unnormalize_action(self, action):
        """反归一化动作"""
        action = action.clone()
        
        trans_min = torch.tensor(TRANS_MIN, dtype=action.dtype, device=action.device)
        trans_max = torch.tensor(TRANS_MAX, dtype=action.dtype, device=action.device)
        max_gripper_width = torch.tensor(MAX_GRIPPER_WIDTH, dtype=action.dtype, device=action.device)
        
        action[..., :3] = (action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
        action[..., -1] = (action[..., -1] + 1) / 2.0 * max_gripper_width
        return action

    def evaluate(self):
        print("Starting evaluation...")
        
        visualizer = AttentionVisualizer(save_dir="./vis")
        episode_idx = 0
        with torch.no_grad():
            for step_idx, data in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                if step_idx < 140:
                    continue
                cloud_coords = data['input_coords_list'].to(self.device)
                cloud_feats = data['input_feats_list'].to(self.device)
                gt_actions = data['action_normalized'].to(self.device)
                
                human_tracks_abs = data.get('human_tracks_abs', None)
                human_tracks_rel = data.get('human_tracks_rel', None)
                robot_tracks_abs = data.get('robot_tracks_abs', None)
                robot_tracks_rel = data.get('robot_tracks_rel', None)
                human_track_lengths = data.get('human_track_lengths', None)
                robot_track_lengths = data.get('robot_track_lengths', None)
                human_semantics = data.get('human_semantics', None)
                robot_semantics = data.get('robot_semantics', None)
                
                robot_total_length = data.get('robot_total_length', None)

                # Move to device
                cloud_feats = cloud_feats.to(self.device)
                cloud_coords = cloud_coords.to(self.device)
                

                human_tracks_abs = human_tracks_abs.to(self.device)
                human_tracks_rel = human_tracks_rel.to(self.device)
                robot_tracks_abs = robot_tracks_abs.to(self.device)
                robot_tracks_rel = robot_tracks_rel.to(self.device)
            
                human_track_lengths = human_track_lengths.to(self.device)
                robot_track_lengths = robot_track_lengths.to(self.device)
                human_semantics = human_semantics.to(self.device)
                robot_semantics = robot_semantics.to(self.device)
                robot_total_length = robot_total_length.to(self.device)
                # save_tracks_and_semantics("./task_0103_user_0555_scene_0023_cfg_0001_BEFORE_task_0103_user_0555_scene_0025_cfg_0001_AFTER", step_idx,
                #                 human_tracks_abs=human_tracks_abs, robot_tracks_abs=robot_tracks_abs,
                #               human_tracks_rel=human_tracks_rel, robot_tracks_rel=robot_tracks_rel,
                #               human_semantics=human_semantics, robot_semantics=robot_semantics,
                #         human_track_lengths=human_track_lengths, robot_track_lengths=robot_track_lengths,robot_total_length=robot_total_length,cloud_feats=cloud_feats,cloud_coords=cloud_coords)
                cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)

                pred_actions = self.policy(
                    cloud=cloud_data, 
                    actions=None, 
                    human_tracks_abs=human_tracks_abs,
                    human_tracks_rel=human_tracks_rel,
                    robot_tracks_abs=robot_tracks_abs,
                    robot_tracks_rel=robot_tracks_rel,
                    human_track_lengths=human_track_lengths,
                    robot_track_lengths=robot_track_lengths,
                    human_semantics=human_semantics,
                    robot_semantics=robot_semantics,
                    robot_total_length=robot_total_length,
                    batch_size=gt_actions.shape[0]
                )
                # print(pred_actions)
            
                if self.args.vis or self.args.save_results:
                    cloud_numpy = None
                    if 'clouds_list' in data and data['clouds_list'] is not None:
                        cloud_numpy = data['clouds_list'][0][0]
                    
                    save_path = os.path.join(self.args.output_dir, f"step_{step_idx}.txt") if self.args.save_results else None
                    self.visualize_prediction(
                        step_idx, cloud_numpy, 
                        pred_actions.squeeze(0), gt_actions.squeeze(0),
                        save_path
                    )        
        
        return avg_metrics

    def visualize_prediction(self, step_idx, cloud_data, pred_actions, gt_actions, save_path=None):
        """可视化预测结果"""
        if not self.args.vis and save_path is None:
            return
        
        pred_actions_unnorm = self.unnormalize_action(pred_actions)
        gt_actions_unnorm = self.unnormalize_action(gt_actions)
        
        if cloud_data is not None:
            if isinstance(cloud_data, torch.Tensor):
                cloud_data = cloud_data.cpu().numpy()
                
            points = cloud_data[:, :3]
            colors = cloud_data[:, 3:] * np.array(IMG_STD) + np.array(IMG_MEAN)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            pred_traj = []
            gt_traj = []
            
            for i in range(min(16, len(pred_actions_unnorm))):
                pred_sphere = o3d.geometry.TriangleMesh.create_sphere(0.01)
                pred_sphere.translate(pred_actions_unnorm[i, :3].cpu().numpy())
                pred_sphere.paint_uniform_color([1, 0, 0])
                pred_traj.append(pred_sphere)
                
                gt_sphere = o3d.geometry.TriangleMesh.create_sphere(0.01)
                gt_sphere.translate(gt_actions_unnorm[i, :3].cpu().numpy())
                gt_sphere.paint_uniform_color([0, 1, 0])
                gt_traj.append(gt_sphere)
            
            if save_path:
                print(f"Visualization for step {step_idx} (pred: red, gt: green)")
            
            if self.args.vis:
                o3d.visualization.draw_geometries([pcd] + pred_traj + gt_traj)


def main(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value
    
    evaluator = OfflineEvaluator(args)
    results = evaluator.evaluate()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline evaluation script for HistRISE')
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    
    parser.add_argument('--num_action', type=int, default=20, help='Number of action steps')
    parser.add_argument('--num_history', type=int, default=5, help='Number of history actions')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size')
    parser.add_argument('--obs_feature_dim', type=int, default=512, help='Observation feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nheads', type=int, default=8, help='Number of heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
    
    parser.add_argument('--max_test_steps', type=int, default=300, help='Maximum number of test steps')
    parser.add_argument('--scene_filter', type=str, default=None, help='Scene to filter')
    parser.add_argument('--seed', type=int, default=233, help='Random seed')
    
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save evaluation results')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for results')
    
    parser.add_argument('--num_targets', type=int, default=4, help='Number of targets')
    parser.add_argument('--num_points', type=int, default=10, help='Number of points per target')
    
    args = parser.parse_args()
    main(vars(args))