# eval_track_encoder_embeddings.py

import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from policy.track.model import TrackEncoder
from dataset.realworld import RealWorldDataset, collate_fn

def compute_dense_embeddings(tracks, track_lengths, encoder, fusion_layer, num_targets=4, num_points=10, window_size=16):
    """
    Compute embeddings for each timestep t (future window_size frames from t)
    
    Args:
        tracks: (batch, T, num_targets, num_points, 3)
        track_lengths: (batch,)
        encoder: TrackEncoder
        fusion_layer: Linear projection layer
        window_size: number of future frames to use for each window
        
    Returns:
        all_embeddings: List of (T_i, num_targets, hidden_dim) for each batch
    """
    batch_size, T, _, _, _ = tracks.shape
    device = tracks.device
    all_batch_embeddings = []
    
    for b in range(batch_size):
        actual_len = int(track_lengths[b].item())
        batch_embeddings = []
        
        # Only compute for timesteps where we have enough frames
        max_start = max(1, actual_len - window_size + 1)
        
        for t in range(max_start):
            # Extract window from t to t+window_size (future 16 frames from t)
            end_t = min(t + window_size, actual_len)
            track_segment = tracks[b, t:end_t]  # (window_len, num_targets, num_points, 3)
            
            window_len = track_segment.shape[0]
            
            # Reshape for encoder: (num_targets, window_len, num_points, 3)
            encoder_input = track_segment.permute(1, 0, 2, 3)
            
            # Create lengths
            lengths = torch.full((num_targets,), window_len, dtype=torch.long, device=device)
            
            # Encode
            tokens = encoder(encoder_input, lengths=lengths)
            # tokens: (num_targets, num_points, num_queries, dim)
            
            # Fusion & pooling
            _, num_points_enc, num_queries, dim = tokens.shape
            tokens = tokens.view(num_targets, num_points_enc * num_queries, dim)
            tokens = fusion_layer(tokens)
            tokens = tokens.mean(dim=1)  # (num_targets, hidden_dim)
            
            batch_embeddings.append(tokens)
        
        # Stack: (T_i, num_targets, hidden_dim)
        if len(batch_embeddings) > 0:
            all_batch_embeddings.append(torch.stack(batch_embeddings, dim=0))
    
    return all_batch_embeddings

    
def visualize_similarity_matrix(embeddings, save_path, window_size=16, title_suffix=""):
    """
    Visualize target-time joint similarity matrix
    
    Args:
        embeddings: (T, num_targets, hidden_dim)
        save_path: path to save figure
        window_size: window size used for embeddings
    """
    T, num_targets, hidden_dim = embeddings.shape
    
    # Permute to (num_targets, T, hidden_dim)
    all_keys = embeddings.permute(1, 0, 2).contiguous()
    
    # Flatten: (num_targets * T, hidden_dim)
    flat_keys = all_keys.view(-1, hidden_dim)
    
    # Normalize for cosine similarity
    flat_keys_norm = F.normalize(flat_keys, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(flat_keys_norm, flat_keys_norm.t()).detach().cpu().numpy()
    
    # Plot
    plt.figure(figsize=(14, 12))
    im = plt.imshow(sim_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='Cosine Similarity')
    
    # Draw grid lines between targets
    for i in range(1, num_targets):
        boundary = i * T - 0.5
        plt.axhline(y=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
        plt.axvline(x=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set ticks
    tick_locs = []
    tick_labels = []
    tick_interval = max(1, T // 10)
    
    for i in range(num_targets):
        start_idx = i * T
        for t in range(0, T, tick_interval):
            tick_locs.append(start_idx + t)
            tick_labels.append(str(t))
    
    plt.xticks(tick_locs, tick_labels, rotation=90, fontsize=8)
    plt.yticks(tick_locs, tick_labels, fontsize=8)
    
    plt.xlabel('Time Step (Target 0 -> Target N-1)', fontsize=11)
    plt.ylabel('Time Step (Target 0 -> Target N-1)', fontsize=11)
    plt.title(f'Target-Time Joint Similarity Matrix (Window Size: {window_size})\n(Block Size: {T}x{T}){title_suffix}', 
              fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_target_similarities(embeddings, save_path, window_size=16, title_suffix=""):
    """
    Visualize temporal similarity within each target
    
    Args:
        embeddings: (T, num_targets, hidden_dim)
        save_path: path to save figure
        window_size: window size used for embeddings
    """
    T, num_targets, hidden_dim = embeddings.shape
    
    fig, axes = plt.subplots(1, num_targets, figsize=(6*num_targets, 5))
    if num_targets == 1:
        axes = [axes]
    
    for target_idx in range(num_targets):
        target_embeds = embeddings[:, target_idx, :]  # (T, hidden_dim)
        
        # Normalize
        target_embeds_norm = F.normalize(target_embeds, p=2, dim=1)
        
        # Compute self-similarity matrix
        sim_matrix = torch.mm(target_embeds_norm, target_embeds_norm.t())
        sim_matrix = sim_matrix.detach().cpu().numpy()
        
        ax = axes[target_idx]
        im = ax.imshow(sim_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Time Step', fontsize=11)
        ax.set_title(f'Target {target_idx} Temporal Similarity\n(Window Size: {window_size})', 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Per-Target Temporal Self-Similarity{title_suffix}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_embeddings_statistics(embeddings_list, save_dir):
    """
    Analyze statistics of embeddings across dataset
    
    Args:
        embeddings_list: List of (T_i, num_targets, hidden_dim) tensors
        save_dir: directory to save statistics
    """
    all_norms = []
    all_target_similarities = []
    all_temporal_similarities = []
    
    for embeddings in embeddings_list:
        T, num_targets, hidden_dim = embeddings.shape
        
        # Compute norms
        norms = torch.norm(embeddings, p=2, dim=-1)  # (T, num_targets)
        all_norms.append(norms.flatten().cpu().numpy())
        
        # Compute cross-target similarity at same timestep
        for t in range(T):
            target_embeds = embeddings[t]  # (num_targets, hidden_dim)
            target_embeds_norm = F.normalize(target_embeds, p=2, dim=1)
            cross_target_sim = torch.mm(target_embeds_norm, target_embeds_norm.t())
            # Extract upper triangle (excluding diagonal)
            mask = torch.triu(torch.ones_like(cross_target_sim), diagonal=1).bool()
            similarities = cross_target_sim[mask].cpu().numpy()
            all_target_similarities.extend(similarities)
        
        # Compute temporal similarity within each target
        for target_idx in range(num_targets):
            target_embeds = embeddings[:, target_idx, :]  # (T, hidden_dim)
            if T > 1:
                target_embeds_norm = F.normalize(target_embeds, p=2, dim=1)
                temporal_sim = torch.mm(target_embeds_norm, target_embeds_norm.t())
                mask = torch.triu(torch.ones_like(temporal_sim), diagonal=1).bool()
                similarities = temporal_sim[mask].cpu().numpy()
                all_temporal_similarities.extend(similarities)
    
    # Aggregate statistics
    all_norms = np.concatenate(all_norms)
    all_target_similarities = np.array(all_target_similarities)
    all_temporal_similarities = np.array(all_temporal_similarities)
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Norm distribution
    axes[0].hist(all_norms, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(all_norms.mean(), color='red', linestyle='--', 
                    label=f'Mean: {all_norms.mean():.3f}')
    axes[0].set_xlabel('L2 Norm', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Embedding Norm Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cross-target similarity
    axes[1].hist(all_target_similarities, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(all_target_similarities.mean(), color='red', linestyle='--',
                    label=f'Mean: {all_target_similarities.mean():.3f}')
    axes[1].set_xlabel('Cosine Similarity', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Cross-Target Similarity (Same Time)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Temporal similarity
    axes[2].hist(all_temporal_similarities, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(all_temporal_similarities.mean(), color='red', linestyle='--',
                    label=f'Mean: {all_temporal_similarities.mean():.3f}')
    axes[2].set_xlabel('Cosine Similarity', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Temporal Similarity (Same Target)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'embedding_statistics.png'), dpi=150)
    plt.close()
    
    # Save text statistics
    with open(os.path.join(save_dir, 'statistics.txt'), 'w') as f:
        f.write("Embedding Statistics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Norm:\n")
        f.write(f"  Mean: {all_norms.mean():.4f}\n")
        f.write(f"  Std:  {all_norms.std():.4f}\n")
        f.write(f"  Min:  {all_norms.min():.4f}\n")
        f.write(f"  Max:  {all_norms.max():.4f}\n\n")
        
        f.write(f"Cross-Target Similarity (Same Time):\n")
        f.write(f"  Mean: {all_target_similarities.mean():.4f}\n")
        f.write(f"  Std:  {all_target_similarities.std():.4f}\n")
        f.write(f"  Min:  {all_target_similarities.min():.4f}\n")
        f.write(f"  Max:  {all_target_similarities.max():.4f}\n\n")
        
        f.write(f"Temporal Similarity (Same Target):\n")
        f.write(f"  Mean: {all_temporal_similarities.mean():.4f}\n")
        f.write(f"  Std:  {all_temporal_similarities.std():.4f}\n")
        f.write(f"  Min:  {all_temporal_similarities.min():.4f}\n")
        f.write(f"  Max:  {all_temporal_similarities.max():.4f}\n")
    
    print(f"\n统计数据已保存到 {save_dir}/statistics.txt")

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"窗口大小: {args.window_size}")
    
    # Load dataset
    print("加载数据集...")
    dataset = RealWorldDataset(
        path=args.data_path,
        split=args.split,
        num_obs=1,
        num_action=args.num_action,
        num_history=args.num_history,
        voxel_size=args.voxel_size,
        aug=False,
        aug_jitter=False,
        with_cloud=False,
        vis=False,
        num_targets=args.num_targets
    )
    if args.scene_filter:
        filtered_indices = []
        for i, data_path in enumerate(dataset.data_paths):
            if args.scene_filter in data_path:
                filtered_indices.append(i)
        
        filtered_indices = filtered_indices[:]
        
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
    
    # Model config
    common_embed_dim = 256
    encoder_config = {
        'num_points': args.num_points,
        'input_dim': 3,
        'patch_size': 4,
        'embed_dim': 256,
        'query_dim': common_embed_dim,
        'num_queries': 1,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1,
        'use_rope': False,
        'use_time_embedding': True,
        'max_seq_len': 2000,
        'output_dim': None
    }
    
    # Load encoder
    print(f"加载模型从 {args.ckpt_path}...")
    encoder = TrackEncoder(**encoder_config).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    # Extract encoder weights
    encoder_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('encoder.'):
            encoder_dict[k[8:]] = v
        elif not k.startswith('decoder.'):
            encoder_dict[k] = v
    
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.eval()
    
    # Fusion layer
    track_output_dim = encoder_config['output_dim'] or encoder_config['query_dim']
    fusion_layer = torch.nn.Linear(track_output_dim, args.track_enc_dim).to(device)
    
    # Try to load fusion layer weights if available
    fusion_dict = {}
    for k, v in state_dict.items():
        if 'human_track_fusion' in k or 'track_fusion' in k:
            new_k = k.split('.')[-1]
            fusion_dict[new_k] = v
    
    if fusion_dict:
        fusion_layer.load_state_dict(fusion_dict, strict=False)
        print("已加载 fusion layer 权重")
    
    fusion_layer.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation
    all_embeddings = []
    num_samples = 0
    
    print("评估中...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            
            tracks = data['human_tracks_rel'].to(device)
            lengths = data['human_track_lengths'].to(device)
            
            batch_size, T, N, _ = tracks.shape
            
            # Reshape: (batch, T, num_targets, num_points, 3)
            tracks = tracks.view(batch_size, T, args.num_targets, args.num_points, 3)
            
            # Compute embeddings with window_size
            batch_embeddings = compute_dense_embeddings(
                tracks, lengths, encoder, fusion_layer, 
                args.num_targets, args.num_points, args.window_size
            )
            
            all_embeddings.extend(batch_embeddings)
            num_samples += batch_size
            
            # Visualize first few samples
            if batch_idx < args.num_vis_samples:
                for i, embeds in enumerate(batch_embeddings):
                    sample_idx = batch_idx * args.batch_size + i
                    
                    # Save embeddings
                    npy_path = os.path.join(args.output_dir, f'embeddings_sample_{sample_idx:04d}.npy')
                    np.save(npy_path, embeds.cpu().numpy())
                    
                    # Visualize similarity matrix
                    fig_path = os.path.join(args.output_dir, 
                                           f'similarity_matrix_sample_{sample_idx:04d}.png')
                    visualize_similarity_matrix(embeds, fig_path, args.window_size,
                                               f'\nSample {sample_idx}')
                    
                    # Visualize per-target temporal similarity
                    fig_path = os.path.join(args.output_dir, 
                                           f'target_similarity_sample_{sample_idx:04d}.png')
                    visualize_target_similarities(embeds, fig_path, args.window_size,
                                                 f'\nSample {sample_idx}')
    
    print(f"\n评估完成 - 总样本数: {num_samples}")
    print(f"已保存 {len(all_embeddings)} 个样本的 embeddings")
    
    # Analyze statistics
    print("\n分析 embedding 统计信息...")
    analyze_embeddings_statistics(all_embeddings, args.output_dir)
    
    print(f"\n所有结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Track Encoder Embeddings')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Encoder checkpoint路径')
    parser.add_argument('--output_dir', type=str, default='eval_embeddings', help='输出目录')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_action', type=int, default=20)
    parser.add_argument('--num_history', type=int, default=5)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--num_targets', type=int, default=4, help='目标物体数量')
    parser.add_argument('--num_points', type=int, default=10, help='每个物体的点数')
    parser.add_argument('--track_enc_dim', type=int, default=128, help='Track encoder输出维度')
    parser.add_argument('--max_batches', type=int, default=-1, help='最大batch数(-1表示全部)')
    parser.add_argument('--num_vis_samples', type=int, default=5, help='可视化样本数')
    parser.add_argument('--scene_filter', type=str, default=None, help='Scene to filter')
    parser.add_argument('--window_size', type=int, default=16, help='未来窗口大小')

    args = parser.parse_args()
    evaluate(args)