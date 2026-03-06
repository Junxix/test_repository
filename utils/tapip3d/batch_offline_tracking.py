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
import json

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
                 device: str = "auto"):
        self.device = self._setup_device(device)
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.tapip3d_checkpoint = tapip3d_checkpoint
        
        self._init_sam2()
        self._init_tapip3d()
        
        self.points = []
        self.labels = []
        self.is_accepting_clicks = True
        self.mask = None
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
        print(f"SAM2 model loaded on {self.device}")
    
    def _init_tapip3d(self):
        self.tapip3d_model = load_model(self.tapip3d_checkpoint)
        self.tapip3d_model.to(self.device)
        self.tapip3d_model.eval()
        
        if hasattr(self.tapip3d_model, "set_eval_mode"):
            self.tapip3d_model.set_eval_mode("raw")
        
        print(f"TAPIP3D model loaded on {self.device}")
    
    def load_data_with_segments(self, input_path: str, target_resolution: Optional[Tuple[int, int]] = None):
        """Load NPZ data and split into before/after/full segments based on human.json"""
        print(f"Loading data from {input_path} ...")
        
        # Find human.json in scene directory
        input_path_obj = Path(input_path)
        scene_dir = input_path_obj.parents[2]  # Go up: tapip3d -> cam_xxx -> scene_dir
        human_json_path = scene_dir / "human.json"
        
        if not human_json_path.exists():
            print(f"Warning: human.json not found at {human_json_path}")
            print("Processing as single segment without splitting")
            return self._load_single_segment(input_path, target_resolution)
        
        # Load human.json to get time info
        with open(human_json_path, 'r') as f:
            time_info = json.load(f)
        
        # Load NPZ data
        data = np.load(input_path)
        
        if 'video' not in data:
            raise ValueError("NPZ file must contain 'video' key.")
        
        video = data['video']
        
        timestamps = np.arange(len(video))
        
        # Find split frame index
        split_idx = int(time_info['robot_start_idx'])
        
        print(f"Split at frame {split_idx}/{len(timestamps)}")
        print(f"Before segment: frames 0-{split_idx-1} ({split_idx} frames)")
        print(f"After segment: frames {split_idx}-{len(timestamps)-1} ({len(timestamps)-split_idx} frames)")
        
        # Create before segment (0 to split_idx-1)
        before_segment = {
            'video': video[:split_idx],
            'timestamps': timestamps[:split_idx],
            'depths': data.get('depths', None)[:split_idx] if 'depths' in data else None,
            'intrinsics': data.get('intrinsics', None)[:split_idx] if 'intrinsics' in data else None,
            'extrinsics': data.get('extrinsics', None)[:split_idx] if 'extrinsics' in data else None,
            'segment_type': 'before',
            'annotation_frame': 0  # Use first frame of before segment
        }
        
        # Create after segment (split_idx to end)
        after_segment = {
            'video': video[split_idx:],
            'timestamps': timestamps[split_idx:],
            'depths': data.get('depths', None)[split_idx:] if 'depths' in data else None,
            'intrinsics': data.get('intrinsics', None)[split_idx:] if 'intrinsics' in data else None,
            'extrinsics': data.get('extrinsics', None)[split_idx:] if 'extrinsics' in data else None,
            'segment_type': 'after',
            'annotation_frame': 0  # Use first frame of after segment (robot_start_time)
        }
        
        # Create full segment (entire video)
        full_segment = {
            'video': video,
            'timestamps': timestamps,
            'depths': data.get('depths', None) if 'depths' in data else None,
            'intrinsics': data.get('intrinsics', None) if 'intrinsics' in data else None,
            'extrinsics': data.get('extrinsics', None) if 'extrinsics' in data else None,
            'segment_type': 'full',
            'annotation_frame': 0  # Use first frame of full video
        }
        
        return before_segment, after_segment, full_segment
    
    def _load_single_segment(self, input_path: str, target_resolution: Optional[Tuple[int, int]] = None):
        """Load data as single segment when human.json is not found"""
        data = np.load(input_path)
        
        if 'video' not in data:
            raise ValueError("NPZ file must contain 'video' key.")
        
        video = data['video']
        timestamps = data.get('timestamps', np.arange(len(video)))
        
        full_segment = {
            'video': video,
            'timestamps': timestamps,
            'depths': data.get('depths', None) if 'depths' in data else None,
            'intrinsics': data.get('intrinsics', None) if 'intrinsics' in data else None,
            'extrinsics': data.get('extrinsics', None) if 'extrinsics' in data else None,
            'segment_type': 'full',
            'annotation_frame': 0
        }
        
        # Return None for before/after segments
        return None, None, full_segment
    
    def load_segment(self, segment_data: Dict, target_resolution: Optional[Tuple[int, int]] = None):
        """Load a single segment into the model"""
        video = segment_data['video']
        
        # Convert to tensor (T, H, W, 3) -> (T, 3, H, W)
        if video.ndim == 4 and video.shape[-1] == 3:
            self.video_data = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        else:
            raise ValueError(f"Video shape {video.shape} not supported.")
        
        # Load depth data
        if segment_data['depths'] is not None:
            self.depth_data = torch.from_numpy(segment_data['depths']).float()
        else:
            T, _, H, W = self.video_data.shape
            self.depth_data = torch.ones(T, H, W) * 5.0
        
        # Load intrinsics
        if segment_data['intrinsics'] is not None:
            self.intrinsics = torch.from_numpy(segment_data['intrinsics']).float()
        else:
            self.intrinsics = torch.from_numpy(self._create_default_intrinsics()).float()
        
        # Load extrinsics
        if segment_data['extrinsics'] is not None:
            self.extrinsics = torch.from_numpy(segment_data['extrinsics']).float()
        else:
            T = self.video_data.shape[0]
            self.extrinsics = torch.eye(4).unsqueeze(0).repeat(T, 1, 1).float()
        
        if target_resolution:
            self._resize_data(target_resolution)
        
        self._move_data_to_device()
        
        return segment_data['annotation_frame'], segment_data['segment_type']
    
    def _create_default_intrinsics(self) -> np.ndarray:
        T, _, H, W = self.video_data.shape
        f = min(H, W) / (2 * np.tan(np.pi / 6))
        intrinsics = np.eye(3)
        intrinsics[0, 0] = f  # fx
        intrinsics[1, 1] = f  # fy
        intrinsics[0, 2] = W / 2  # cx
        intrinsics[1, 2] = H / 2  # cy
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
        print("Set SAM2 state for new target")
        
        if self.video_dir is not None:
            self.inference_state = self.sam2_predictor.init_state(video_path=self.video_dir)
            print("SAM2 state initialized")
        else:
            raise ValueError("video_dir is None, please run prepare_sam2_video() first.")
    
    def interactive_segment(self, frame_idx: int = 0, target_id: int = 1) -> Optional[torch.Tensor]:
        self._reset_sam2_state_for_new_target()
        
        self.points = []
        self.labels = []
        self.is_accepting_clicks = True
        self.mask = None
        
        frame = self.video_data[frame_idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Target {target_id} - Frame {frame_idx}")
        plt.imshow(frame)
        
        onclick_callback = functools.partial(
            self._on_click, 
            frame_idx=frame_idx, 
            target_id=target_id
        )
        onkey_callback = functools.partial(
            self._on_key, 
            frame_idx=frame_idx, 
            target_id=target_id
        )
        
        canvas = plt.gcf().canvas
        canvas.mpl_connect('button_press_event', onclick_callback)
        canvas.mpl_connect('key_press_event', onkey_callback)
        
        plt.show()
        
        if self.mask is not None:
            print(f"Target {target_id} annotation completed with {len(self.points)} points.")
            return torch.from_numpy(self.mask.astype(bool))
        else:
            print(f"Target {target_id} annotation skipped.")
            return None
    
    def _on_click(self, event, frame_idx: int, target_id: int):
        if not self.is_accepting_clicks or event.xdata is None or event.ydata is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        self.is_accepting_clicks = False
        
        if event.button in [1, 3]: 
            label = 1 if event.button == 1 else 0
            self.points.append([x, y])
            self.labels.append(label)
            print(f"Add point: {[x, y]} with label {label}")
            
            self._update_sam2_prediction(frame_idx, target_id)
            
        self.is_accepting_clicks = True
    
    def _on_key(self, event, frame_idx: int, target_id: int):
        if event.key == 'z' and len(self.points) > 0:
            removed_point = self.points.pop()
            removed_label = self.labels.pop()
            print(f"Delete point: {removed_point} with label {removed_label}")
            
            if len(self.points) > 0:
                self._update_sam2_prediction(frame_idx, target_id)
            else:
                self._clear_display()
    
    def _update_sam2_prediction(self, frame_idx: int, target_id: int):
        if len(self.points) == 0:
            return
            
        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=np.array(self.points),
            labels=np.array(self.labels)
        )
        
        self._clear_display()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]
            self.mask = mask
            self._show_points(np.array(self.points), np.array(self.labels))
            self._show_mask(mask, obj_id=target_id)
    
    def _clear_display(self):
        ax = plt.gca()
        images = ax.images
        if len(images) > 1:
            for img in images[1:]:
                img.remove()
        for collection in ax.collections:
            collection.remove()
        plt.draw()
    
    def _show_points(self, coords, labels, marker_size=200):
        ax = plt.gca()
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                      color='green', marker='*', s=marker_size, 
                      edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                      color='red', marker='*', s=marker_size, 
                      edgecolor='white', linewidth=1.25)
    
    def _show_mask(self, mask, obj_id=None):
        ax = plt.gca()
        
        if obj_id is not None:
            color = np.array([*plt.get_cmap("tab10")(obj_id)[:3], 0.6])
        else:
            color = np.array([1, 0, 0, 0.6])
            
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image, alpha=0.6)
        plt.draw()

    # ==========================
    # MODIFIED FUNCTION START
    # ==========================
    def track_with_tapip3d(self, mask: torch.Tensor, grid_size: int = 0, 
                            num_iters: int = 6, vis_threshold: float = 0.9,
                            frame_batch_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inference in frame batches, using last frame results as next batch's query.
        """
        mask = mask.to(self.device)
        
        # Generate initial query points from mask at frame 0
        query_points = self._generate_query_points_from_mask(mask)
        
        n_mask_points = query_points.shape[1]
        T = self.video_data.shape[0]
        
        H, W = self.video_data.shape[2:]
        self.tapip3d_model.set_image_size((H, W))
        
        print(f"Tracking {n_mask_points} points across {T} frames in batches of {frame_batch_size}...")
        
        all_coords = []
        all_visibs = []
        
        current_queries = query_points.squeeze(0)  # (N, 4): [t, x, y, z]
        
        # Split frames into batches
        for batch_idx, start_frame in enumerate(tqdm(range(0, T, frame_batch_size), desc="Frame Batches")):
            end_frame = min(start_frame + frame_batch_size, T)
            batch_length = end_frame - start_frame
            
            # Slice video data for this batch
            batch_video = self.video_data[start_frame:end_frame]
            batch_depth = self.depth_data[start_frame:end_frame]
            batch_intrinsics = self.intrinsics[start_frame:end_frame]
            batch_extrinsics = self.extrinsics[start_frame:end_frame]
            
            # Set query time to 0 (start of this batch)
            current_queries[:, 0] = 0
            
            with torch.no_grad():
                chunk_coords, chunk_visibs = inference(
                    model=self.tapip3d_model,
                    video=batch_video,
                    depths=batch_depth,
                    intrinsics=batch_intrinsics,
                    extrinsics=batch_extrinsics,
                    query_point=current_queries,
                    num_iters=num_iters,
                    grid_size=0,
                    vis_threshold=vis_threshold
                )
            
            # Debug: Print shapes on first batch
            if batch_idx == 0:
                print(f"chunk_coords shape: {chunk_coords.shape}")
                print(f"chunk_visibs shape: {chunk_visibs.shape}")
                print(f"n_mask_points: {n_mask_points}")
            
            # Save results (move to CPU to save GPU memory)
            all_coords.append(chunk_coords.cpu())
            all_visibs.append(chunk_visibs.cpu())
            
            # Use last frame's 3D coordinates as next batch's query
            if end_frame < T:  # Not the last batch
                # chunk_coords expected shape: (N, T_batch, 3) or (T_batch, N, 3)
                if chunk_coords.shape[0] == n_mask_points:
                    # Shape: (N, T_batch, 3)
                    last_frame_coords = chunk_coords[:, -1, :]  # (N, 3)
                elif chunk_coords.shape[1] == n_mask_points:
                    # Shape: (T_batch, N, 3)
                    last_frame_coords = chunk_coords[-1, :, :]  # (N, 3)
                else:
                    raise ValueError(f"Unexpected chunk_coords shape: {chunk_coords.shape}, expected first or second dim to be {n_mask_points}")
                
                # Construct new query: [t=0, x, y, z]
                current_queries = torch.cat([
                    torch.zeros(n_mask_points, 1, device=self.device),
                    last_frame_coords
                ], dim=1)
            
            torch.cuda.empty_cache()
        
        # Concatenate along time dimension
        # Determine correct concatenation dimension based on shape
        if all_coords[0].shape[0] == n_mask_points:
            concat_dim = 1  # Shape: (N, T, 3)
        else:
            concat_dim = 0  # Shape: (T, N, 3)
        
        final_coords = torch.cat(all_coords, dim=concat_dim).to(self.device)
        final_visibs = torch.cat(all_visibs, dim=concat_dim if concat_dim == 0 else 1).to(self.device)
        
        print(f"Final coords shape: {final_coords.shape}")
        print(f"Final visibs shape: {final_visibs.shape}")
        
        return final_coords, final_visibs
    # ==========================
    # MODIFIED FUNCTION END
    # ==========================
    
    def _generate_query_points_from_mask(self, mask: torch.Tensor, num_points: int = 150) -> torch.Tensor:
        mask_indices = torch.nonzero(mask, as_tuple=False)
        
        if len(mask_indices) == 0:
            raise ValueError("Mask has no positive pixels.")
        
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
            torch.zeros(len(world_coords), 1, device=self.device),  # t=0
            world_coords
        ], dim=1).unsqueeze(0)
        
        return query_points
    
    def save_results(self, coords: torch.Tensor, visibs: torch.Tensor, 
                    mask: torch.Tensor, target_name: str, output_dir: str = "./results"):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        coords_path = Path(output_dir) / f"3d_tracks_{target_name}.npy"
        visibs_path = Path(output_dir) / f"visibility_{target_name}.npy"
        mask_path = Path(output_dir) / f"mask_{target_name}.png"
        
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        visibs_np = visibs.cpu().numpy() if torch.is_tensor(visibs) else visibs
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        
        np.save(coords_path, coords_np)
        np.save(visibs_path, visibs_np)
        
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_path)
        
        result_path = Path(output_dir) / f"complete_result_{target_name}.npz"
        np.savez(
            result_path,
            video=(self.video_data.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)[:1],
            depths=self.depth_data.cpu().numpy()[:1],
            intrinsics=self.intrinsics.cpu().numpy()[:1],
            extrinsics=self.extrinsics.cpu().numpy()[:1],
            coords=coords_np,
            visibs=visibs_np,
            mask=mask_np
        )

        return result_path
    
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="SAM2 + TAPIP3D for segmented 3D tracking")
    parser.add_argument("--input", required=True, help="Input NPZ file path")
    parser.add_argument("--sam2_checkpoint", required=True, help="SAM2 model checkpoint path")
    parser.add_argument("--sam2_config", required=True, help="SAM2 config file path")
    parser.add_argument("--tapip3d_checkpoint", required=True, help="TAPIP3D model checkpoint path")
    parser.add_argument("--output_dir", default="./sam2_tapip3d_results_offline", help="Output directory")
    parser.add_argument("--target_resolution", type=int, nargs=2, help="Target resolution (H W)")
    parser.add_argument("--num_targets_per_segment", type=int, default=2, help="Targets per segment")
    parser.add_argument("--grid_size", type=int, default=0, help="Grid points for tracking")
    parser.add_argument("--num_iters", type=int, default=6, help="Iterations for tracking")
    
    # Added argument for batch size
    parser.add_argument("--query_batch_size", type=int, default=64, help="Batch size for tracking query points to save GPU memory")
    parser.add_argument("--frame_batch_size", type=int, default=100, help="Batch size for frames")

    parser.add_argument("--device", default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    
    args = parser.parse_args()
    
    setup_logger()
    
    integrator = ImprovedSAM2TAPIP3DIntegration(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        tapip3d_checkpoint=args.tapip3d_checkpoint,
        device=args.device
    )
    
    target_resolution = tuple(args.target_resolution) if args.target_resolution else None
    
    # Load and split data into before, after, and full segments
    before_segment, after_segment, full_segment = integrator.load_data_with_segments(
        args.input, target_resolution
    )
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store results for later concatenation
    # Structure: {'before': {0: res, 1: res}, 'after': {0: res, 1: res}}
    stored_results = {}
    
    # Process order: before, after ONLY (Skipping Full for computation)
    segments_to_process = []
    if before_segment is not None:
        segments_to_process.append(('before', before_segment))
    if after_segment is not None:
        segments_to_process.append(('after', after_segment))
    
    for segment_name, segment in segments_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {segment_name.upper()} segment")
        print(f"Frames: {len(segment['video'])}")
        print(f"{'='*60}")
        
        # Initialize storage for this segment
        stored_results[segment_name] = {}
        
        # Load segment data
        annotation_frame, segment_type = integrator.load_segment(segment, target_resolution)
        
        # Prepare video frames for SAM2
        temp_dir = integrator.prepare_sam2_video()
        
        # Process targets for this segment
        for i in range(args.num_targets_per_segment):
            target_name = f"target_{segment_name}_{i+1}"
            
            print(f"\n--- {target_name} ---")
            
            # Interactive segmentation
            mask = integrator.interactive_segment(
                frame_idx=annotation_frame, 
                target_id=i+1
            )
            
            if mask is None:
                print(f"{target_name} skipped")
                continue
            
            # Erode mask
            kernel_size = 5
            mask = mask.float().unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
            mask = F.conv2d(F.pad(mask, (2, 2, 2, 2)), kernel)
            mask = (mask == kernel_size * kernel_size).float().squeeze()
            
            # Track with TAPIP3D - USING BATCHING
            coords, visibs = integrator.track_with_tapip3d(
                mask=mask,
                grid_size=args.grid_size,
                num_iters=args.num_iters,
                vis_threshold=0.9,
                frame_batch_size=args.frame_batch_size  # 新参数
            )
            
            # Save results
            result_path = integrator.save_results(
                coords, visibs, mask, target_name, args.output_dir
            )
            
            # Store results for merging (keep on GPU or CPU as needed, moving to CPU for safety)
            stored_results[segment_name][i] = {
                'coords': coords.cpu(),
                'visibs': visibs.cpu(),
                'mask': mask.cpu()
            }
            
            print(f"{target_name} saved to {result_path}")
        
        # Cleanup temp dir
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        
        # Clear GPU memory between segments
        integrator.video_data = None
        integrator.depth_data = None
        integrator.intrinsics = None
        integrator.extrinsics = None
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*60}")
    print(f"All processing completed!")
    print(f"{'='*60}")
    
    integrator.cleanup()
    return 0


if __name__ == "__main__":
    exit(main())