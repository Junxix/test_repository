import os
import torch
import argparse
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME
import matplotlib
matplotlib.use('Agg')  # 使用非交互式backend

import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict

from policy import RISE
from dataset.realworld import RealWorldDataset, collate_fn
from utils.constants import *
from utils.training import set_seed
from utils.transformation import xyz_rot_transform, rotation_transform


default_args = edict({
    "ckpt": None,
    "data_path": "data/collect_pens",
    "split": "val",  # or "train"
    "num_action": 20,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "batch_size": 1,
    "num_workers": 0,
    "seed": 233,
    "num_samples": 10,  # number of samples to visualize
    "save_metrics": True,
    "save_dir": "eval_results"
})


def unnormalize_action(action):
    """反归一化动作"""
    action = action.clone()
    
    trans_min = torch.tensor(TRANS_MIN, dtype=action.dtype, device=action.device)
    trans_max = torch.tensor(TRANS_MAX, dtype=action.dtype, device=action.device)
    max_gripper_width = torch.tensor(MAX_GRIPPER_WIDTH, dtype=action.dtype, device=action.device)
    
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    action[..., -1] = (action[..., -1] + 1) / 2.0 * max_gripper_width
    return action



def visualize_prediction(cloud_data, pred_actions, gt_actions):
    pred_actions_unnorm = unnormalize_action(pred_actions)
    gt_actions_unnorm = unnormalize_action(gt_actions)
    
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
        
        o3d.visualization.draw_geometries([pcd] + pred_traj + gt_traj)

def evaluate(args_override):
    # Load args
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.save_metrics and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load dataset
    print("加载数据集...")
    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'train',
        num_obs = 1,
        num_action = args.num_action,
        num_history = args.num_history,
        voxel_size = args.voxel_size,
        aug = args.aug,
        aug_jitter = args.aug_jitter, 
        with_cloud = True,
        vis = args.vis_data,
        num_targets = 2
    )
    # 过滤数据集
    if args.scene_filter:
        filtered_indices = []
        for i, data_path in enumerate(dataset.data_paths):
            if args.scene_filter in data_path:
                filtered_indices.append(i)
        
        filtered_indices = filtered_indices[:args.max_test_steps]
        
        dataset.data_paths = [dataset.data_paths[i] for i in filtered_indices]
        dataset.cam_ids = [dataset.cam_ids[i] for i in filtered_indices]
        dataset.calib_timestamp = [dataset.calib_timestamp[i] for i in filtered_indices]
        dataset.obs_frame_ids = [dataset.obs_frame_ids[i] for i in filtered_indices]
        dataset.action_frame_ids = [dataset.action_frame_ids[i] for i in filtered_indices]
        dataset.history_frame_ids = [dataset.history_frame_ids[i] for i in filtered_indices]
        dataset.track_indices = [dataset.track_indices[i] for i in filtered_indices]
        
        print(f"Filtered dataset to {len(dataset.data_paths)} samples from {args.scene_filter}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False
    )
    

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

    # Load model
    print("加载模型...")
    policy = RISE(
        num_action=args.num_action,
        input_dim=6,  # sinput feature dim
        obs_feature_dim=args.obs_feature_dim,
        action_dim=10,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_attn_layers=4,
        dropout=args.dropout,
        track_config = track_config,
        num_targets=2,  # or from args
        num_points=10
    ).to(device)
    assert args.ckpt is not None, "请提供checkpoint路径"
    
    policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    print(f"已加载checkpoint: {args.ckpt}")
    
    # Evaluation
    policy.eval()
    

    print("开始评估...")
    with torch.inference_mode():
        for sample_idx, data in enumerate(tqdm(dataloader)):
            if sample_idx <= args.num_samples:
                continue

            cloud_coords = data['input_coords_list'].to(device)
            cloud_feats = data['input_feats_list'].to(device)
            gt_actions = data['action_normalized'].to(device)      

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
            cloud_feats = cloud_feats.to(device)
            cloud_coords = cloud_coords.to(device)
            

            human_tracks_abs = human_tracks_abs.to(device)
            human_tracks_rel = human_tracks_rel.to(device)
            robot_tracks_abs = robot_tracks_abs.to(device)
            robot_tracks_rel = robot_tracks_rel.to(device)
        
            human_track_lengths = human_track_lengths.to(device)
            robot_track_lengths = robot_track_lengths.to(device)
            human_semantics = human_semantics.to(device)
            robot_semantics = robot_semantics.to(device)
            robot_total_length = robot_total_length.to(device)


            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            sinput_feats = cloud_data.F
            sinput_coords = cloud_data.C
            # Predict
            
            pred_actions = policy(
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
                batch_size=1
            )
        
            # pred_actions = pred_actions[0].cpu().numpy()
            cloud_numpy = data['clouds_list'][0][0]
                    
            visualize_prediction(cloud_numpy, pred_actions.squeeze(0), gt_actions.squeeze(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--data_path', type=str, required=True, help='data path')
    parser.add_argument('--split', type=str, default='val', help='data split: train/val/all')
    parser.add_argument('--num_action', type=int, default=20, help='number of action steps')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='voxel size')
    parser.add_argument('--obs_feature_dim', type=int, default=512, help='observation feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--nheads', type=int, default=8, help='number of heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--num_samples', type=int, default=0, help='number of samples to visualize')
    parser.add_argument('--seed', type=int, default=233, help='random seed')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='directory to save results')
    parser.add_argument('--scene_filter', type=str, default=None, help='Scene to filter')
    parser.add_argument('--max_test_steps', type=int, default=3000, help='Maximum number of test steps')
    parser.add_argument('--num_history', action = 'store', type = int, help = 'number of history actions', required = False, default = 5)
    parser.add_argument('--aug', action = 'store_true', help = 'whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action = 'store_true', help = 'whether to add color jitter augmentation')
    parser.add_argument('--vis_data', action = 'store_true', help = 'whether to visualize the input data and ground truth actions.')

    evaluate(vars(parser.parse_args()))