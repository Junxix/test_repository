#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import argparse
from pathlib import Path
import logging
import open3d as o3d
import sys
sys.path.append('/home/jingjing/workspace/su1/TAPIP3D/')

from utils.common_utils import batch_unproject, setup_logger
import matplotlib.pyplot as plt

def create_pointcloud_from_rgbd_data(rgb_img, depth_img, intrinsics_matrix, extrinsics_matrix, 
                                   depth_scale=1.0, max_depth=2.0, min_depth=0.1):
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    depth = depth_img.flatten()
    
    scaled_depth = depth * depth_scale
    valid_mask = (depth > 0) & (~np.isinf(depth)) & (scaled_depth >= min_depth) & (scaled_depth <= max_depth)
    
    intrinsics_matrix = np.array([
        [914.81945801,   0.        , 630.63891602],
       [  0.        , 913.88464355, 352.51571655],
        [  0.        ,   0.        ,   1.        ]
    ])

    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = scaled_depth[valid_mask]
    
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    camera_coords = np.column_stack((x, y, z, np.ones(len(x))))
    inv_extrinsics = np.linalg.inv(extrinsics_matrix)
    world_coords = (inv_extrinsics @ camera_coords.T).T
    points_3d = world_coords[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if rgb_img is not None:
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0
            
        rgb_flat = rgb_img.reshape(-1, 3)
        colors_valid = rgb_flat[valid_mask]
        
        colors_valid = np.clip(colors_valid, 0.0, 1.0)
        
        pcd.colors = o3d.utility.Vector3dVector(colors_valid)

    return pcd

def create_trajectory_lines(coords, visibs, colors=None):
    T, N, _ = coords.shape
    
    if colors is None:
        cmap = plt.cm.rainbow
        colors = [cmap(i / N)[:3] for i in range(N)]
    
    line_sets = []
    
    for n in range(N):
        valid_mask = visibs[:, n] if visibs is not None else np.ones(T, dtype=bool)
        valid_coords = coords[valid_mask, n]
        valid_coords = coords[:, n]
        
        if len(valid_coords) < 2:
            continue
            
        lines = [[i, i+1] for i in range(len(valid_coords)-1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(valid_coords)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        color = colors[n]
        line_colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        line_sets.append(line_set)
    
    return line_sets

def create_trajectory_spheres(coords, visibs, frame_idx=0, colors=None, radius=0.02):
    N = coords.shape[1]
    
    if colors is None:
        cmap = plt.cm.rainbow
        colors = [cmap(i / N)[:3] for i in range(N)]
    
    spheres = []
    
    for n in range(N):
        if frame_idx < len(visibs) and (visibs is None or visibs[frame_idx, n]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(coords[frame_idx, n])
            
            color = colors[n]
            sphere.paint_uniform_color(color)
            spheres.append(sphere)
    
    return spheres


def visualize_trajectory_pointcloud_open3d(npz_path: str, max_points: int = 50000):
    setup_logger()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {npz_path}")
    data = np.load(npz_path)
    
    video = data['video']  # (T, H, W, C)
    depths = data['depths']  # (T, H, W)
    intrinsics = data['intrinsics']  # (T, 3, 3)
    extrinsics = data['extrinsics']  # (T, 4, 4)
    coords = data['coords'] # (T, N, 3)
    visibs = data['visibs']   # (T, N)
    
    T, N, _ = coords.shape
    logger.info(f"Data shape: {T} frames, {N} trajectory points")
    logger.info(f"Video shape: {video.shape}")
    
    first_frame_rgb = video[0]  # (H, W, C)
    first_frame_depth = depths[0]  # (H, W)
    first_frame_intrinsics = intrinsics[0]  # (3, 3)
    first_frame_extrinsics = extrinsics[0]  # (4, 4)

    logger.info("Generating background point cloud from first frame...")
    
    point_cloud = create_pointcloud_from_rgbd_data(
        first_frame_rgb, 
        first_frame_depth, 
        first_frame_intrinsics, 
        first_frame_extrinsics
    )
    
    if len(point_cloud.points) > max_points:
        logger.info(f"Downsampling point cloud from {len(point_cloud.points)} to {max_points} points")
        indices = np.random.choice(len(point_cloud.points), max_points, replace=False)
        point_cloud = point_cloud.select_by_index(indices)
    
    logger.info(f"Final point cloud has {len(point_cloud.points)} points")
    
    logger.info("Processing trajectories...")
    
    if visibs.dtype != np.bool_:
        visibs = visibs >= 0
    
    trajectory_lines = create_trajectory_lines(coords, visibs)
    
    trajectory_spheres = create_trajectory_spheres(coords, visibs, frame_idx=0, radius=0.02)
    
    logger.info("Starting visualization...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="TAPIP3D Trajectory Visualization", width=1200, height=800)
    
    vis.add_geometry(point_cloud)
    
    for line_set in trajectory_lines:
        vis.add_geometry(line_set)
    
    for sphere in trajectory_spheres:
        vis.add_geometry(sphere)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.line_width = 3.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_up([0, -1, 0])
    
    logger.info("Visualization ready! Controls:")
    logger.info("  - Mouse: Rotate view")
    logger.info("  - Mouse wheel: Zoom")
    logger.info("  - Ctrl+Mouse: Pan")
    logger.info("  - Press 'Q' or close window to quit")
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize TAPIP3D trajectory results with Open3D")
    parser.add_argument(
        "npz_path", 
        type=str, 
        help="Path to the inference result .npz file"
    )
    parser.add_argument(
        "--max-points", 
        type=int, 
        default=50000, 
        help="Maximum number of points to display (default: 50000)"
    )
    
    args = parser.parse_args()
    
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File {npz_path} does not exist!")
        return
    
    if not npz_path.suffix == ".npz":
        print(f"Error: Expected .npz file, got {npz_path.suffix}")
        return
    
    visualize_trajectory_pointcloud_open3d(str(npz_path), args.max_points)


if __name__ == "__main__":
    main()