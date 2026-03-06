#!/usr/bin/env python3

import os
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def natural_sort_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def load_images_from_directory(image_dir, supported_extensions=('.png', '.jpg', '.jpeg')):
    image_dir = Path(image_dir)
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    image_files.sort(key=lambda x: natural_sort_key(x.name))
    
    return image_files

def load_depth_image(depth_path, depth_scale=1000.0):
    if depth_path.suffix.lower() in ['.png', '.tiff', '.tif']:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"No depth data found in: {depth_path}")
        depth = depth.astype(np.float32) / depth_scale
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"No depth data found in: {depth_path}")
        depth = depth.astype(np.float32)
    
    return depth

def create_tapip3d_npz(color_dir, depth_dir, output_path, intrinsics_matrix, extrinsics_matrix, depth_scale=1000.0):
    color_files = load_images_from_directory(color_dir)
    depth_files = load_images_from_directory(depth_dir, ('.png', '.tiff', '.tif'))
    
    if len(color_files) != len(depth_files):
        print(f"RGB samples: {len(color_files)}, Depth samples: {len(depth_files)}")
        min_count = min(len(color_files), len(depth_files))
        color_files = color_files[:min_count]
        depth_files = depth_files[:min_count]
    
    T = len(color_files)
    
    first_color = cv2.imread(str(color_files[0]))
    first_depth = load_depth_image(depth_files[0], depth_scale)
    
    if first_color is None:
        raise ValueError(f"cannot read first RGB image: {color_files[0]}")
    
    H, W = first_color.shape[:2]
    if first_depth.shape != (H, W):
        print(f"depth image size {first_depth.shape} does not match RGB image size {(H, W)}")
    
    video = np.zeros((T, H, W, 3), dtype=np.uint8)
    depths = np.zeros((T, H, W), dtype=np.float32)
    
    intrinsics = np.tile(intrinsics_matrix[None, :, :], (T, 1, 1)).astype(np.float32)
    extrinsics = np.tile(extrinsics_matrix[None, :, :], (T, 1, 1)).astype(np.float32)
    
    for i in tqdm(range(T), desc="Processing frames"):
        color = cv2.imread(str(color_files[i]))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        video[i] = color
        depth = load_depth_image(depth_files[i], depth_scale)
        
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        
        depths[i] = depth
    
    print(f"Saving to {output_path}")
    np.savez(
        output_path,
        video=video,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )

def main():
    parser = argparse.ArgumentParser(description="convert RGB-D images to TAPIP3D npz format")
    parser.add_argument("--color_dir", required=True, help="rgb dir")
    parser.add_argument("--depth_dir", required=True, help="depth dir")
    parser.add_argument("--output", required=True, help="output npz file path")
    parser.add_argument("--depth_scale", type=float, default=1000.0, 
                       help="depth image scale factor, e.g., 1000.0 if depth is in millimeters")
    
    args = parser.parse_args()
    
    intrinsics_matrix = np.array([
        [922.37457275,   0.        , 637.55419922],
        [  0.        , 922.46069336, 368.37557983],
        [  0.        ,   0.        ,   1.        ]
    ])
    extrinsics_matrix = np.eye(4) 
    
    print("start processing...")
    print(f"rgb: {args.color_dir}")
    print(f"depth: {args.depth_dir}")
    print(f"output: {args.output}")
    
    create_tapip3d_npz(
        color_dir=args.color_dir,
        depth_dir=args.depth_dir,
        output_path=args.output,
        intrinsics_matrix=intrinsics_matrix,
        extrinsics_matrix=extrinsics_matrix,
        depth_scale=args.depth_scale
    )

if __name__ == "__main__":
    main()