# train_human_track_encoder_mae.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
import torch.distributed as dist
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy.track.model import TrackEncoder
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss


class AdaLNBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 4 * embed_dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(condition).chunk(4, dim=1)
        x_norm1 = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        x_norm2 = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        return x

    def _modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MAETrackDecoder(nn.Module):
    def __init__(self, embed_dim=256, patch_size=4, num_points=10, output_dim=3,
                 num_layers=4, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_points = num_points
        self.output_dim = output_dim
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        nn.init.xavier_uniform_(self.mask_token)
        self.pos_embed = nn.Parameter(torch.randn(1, 2000, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.decoder_layers = nn.ModuleList([
            AdaLNBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.output_proj = nn.Linear(embed_dim, patch_size * output_dim)
    
    def forward(self, patches, mask_indices, global_condition):
        batch_size, num_patches, _, _ = patches.shape
        decoder_input = patches.clone()
        mask_token_expanded = self.mask_token.expand(batch_size, num_patches, self.num_points, -1)
        mask_bool = mask_indices.unsqueeze(-1).unsqueeze(-1)
        decoder_input = torch.where(mask_bool, mask_token_expanded, decoder_input)
        decoder_input = decoder_input + self.pos_embed[:, :num_patches, :, :]
        
        all_point_outputs = []
        for point_idx in range(self.num_points):
            seq_tokens = decoder_input[:, :, point_idx, :]
            cond = global_condition[:, point_idx, :]
            for layer in self.decoder_layers:
                seq_tokens = layer(seq_tokens, cond)
            seq_tokens = self.final_norm(seq_tokens)
            point_out = self.output_proj(seq_tokens)
            point_out = point_out.view(batch_size, num_patches, self.patch_size, self.output_dim)
            all_point_outputs.append(point_out)
        
        reconstructed = torch.stack(all_point_outputs, dim=2)
        return reconstructed


class TrackEncoderMAE(nn.Module):
    def __init__(self, encoder_config, decoder_config, mask_ratio=0.75, mask_strategy='random'):
        super().__init__()
        self.encoder = TrackEncoder(**encoder_config)
        self.decoder = MAETrackDecoder(**decoder_config)
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.patch_size = encoder_config['patch_size']
        
    def random_masking(self, num_patches, lengths, device):
        batch_size = lengths.size(0)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        visible_indices = []
        
        for b in range(batch_size):
            actual_len = lengths[b].item()
            num_masked = int(actual_len * self.mask_ratio)
            perm = torch.randperm(actual_len, device=device)
            masked_idx = perm[:num_masked]
            visible_idx = perm[num_masked:]
            mask[b, masked_idx] = True
            visible_indices.append(visible_idx)
        
        return mask, visible_indices
    
    def forward(self, tracks, lengths):
        batch_size, seq_len, num_points, _ = tracks.shape
        device = tracks.device
        
        full_patches, patch_lengths = self.encoder.encoder.point_patch_embed(tracks, lengths)
        num_patches = full_patches.size(1)
        
        mask, visible_indices = self.random_masking(num_patches, patch_lengths, device)
        
        visible_patches_list = []
        for b in range(batch_size):
            visible_patches_list.append(full_patches[b, visible_indices[b]])
        
        max_visible = max(max(vp.size(0) for vp in visible_patches_list), 1)
        
        enc_input = torch.zeros(batch_size, max_visible, num_points, full_patches.size(-1),
                               device=device, dtype=full_patches.dtype)
        enc_input_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b, vp in enumerate(visible_patches_list):
            if vp.size(0) > 0:
                enc_input[b, :vp.size(0)] = vp
                enc_input_lengths[b] = vp.size(0)
            
        if self.encoder.encoder.use_time_embedding:
            input_positions = torch.zeros(batch_size, max_visible, dtype=torch.long, device=device)
            for b in range(batch_size):
                if len(visible_indices[b]) > 0:
                    input_positions[b, :len(visible_indices[b])] = visible_indices[b]
        else:
            input_positions = None
            
        queries = self.encoder.encoder.queries.expand(batch_size, -1, -1)
        all_point_feats = []
        
        for point_idx in range(num_points):
            point_patches = enc_input[:, :, point_idx, :]
            point_queries = queries.clone()
            point_mask = torch.arange(max_visible, device=device)[None, :] < enc_input_lengths[:, None]
            
            point_out = self.encoder.encoder.cross_attention_block(
                point_queries, point_patches, point_mask, input_positions
            )
            
            if self.encoder.encoder.num_queries == 1:
                point_out = self.encoder.encoder.linear_transform(point_out)
            else:
                if self.encoder.encoder.self_attention_blocks is not None:
                    for blk in self.encoder.encoder.self_attention_blocks:
                        point_out = blk(point_out)
                
            all_point_feats.append(point_out)
            
        encoder_output = torch.stack(all_point_feats, dim=2)
        encoder_output = self.encoder.encoder.final_norm(encoder_output)
        
        if encoder_output.size(1) == 1:
            global_condition = encoder_output.squeeze(1)
        else:
            global_condition = encoder_output.mean(dim=1)

        reconstructed = self.decoder(full_patches, mask, global_condition)
        
        target_len = num_patches * self.patch_size
        curr_len = tracks.shape[1]
        
        if curr_len < target_len:
            padding_len = target_len - curr_len
            padding = torch.zeros(batch_size, padding_len, num_points, 3, device=device, dtype=tracks.dtype)
            tracks_padded = torch.cat([tracks, padding], dim=1)
        else:
            tracks_padded = tracks[:, :target_len]
            
        original_tracks_patched = tracks_padded.view(
            batch_size, num_patches, self.patch_size, num_points, 3
        ).permute(0, 1, 3, 2, 4)
        
        loss_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        loss = F.mse_loss(
            reconstructed * loss_mask,
            original_tracks_patched * loss_mask,
            reduction='sum'
        ) / (loss_mask.sum() + 1e-8)
        
        return loss, reconstructed, mask


# ==========================================
# Sliding Window Extraction
# ==========================================

def extract_sliding_windows(tracks, lengths, window_size, stride_min, stride_max, num_targets, num_points, device):
    """
    Extract sliding windows from human tracks with random stride.
    
    Args:
        tracks: (B, T, num_targets, num_points, 3)
        lengths: (B,)
        window_size: int, window length (e.g., 16)
        stride_min: int, minimum stride (e.g., 1)
        stride_max: int, maximum stride (e.g., 30)
        num_targets: int
        num_points: int
        device: torch device
    
    Returns:
        all_windows: (N_total, window_size, num_points, 3)
        all_lengths: (N_total,)
    """
    batch_size = tracks.shape[0]
    
    all_windows = []
    all_lengths = []
    
    for b in range(batch_size):
        actual_len = lengths[b].item()
        
        for target_idx in range(num_targets):
            target_track = tracks[b, :actual_len, target_idx, :, :]  # (actual_len, num_points, 3)
            
            if actual_len <= window_size:
                # sequence too short, use entire sequence
                all_windows.append(target_track)
                all_lengths.append(actual_len)
            else:
                # sliding window with random stride
                start = 0
                while start + window_size <= actual_len:
                    window = target_track[start:start + window_size]  # (window_size, num_points, 3)
                    all_windows.append(window)
                    all_lengths.append(window_size)
                    
                    # random stride for next window
                    stride = torch.randint(stride_min, stride_max + 1, (1,)).item()
                    start += stride
                
                # add last window if not covered
                if start < actual_len and (actual_len - window_size) > (start - stride):
                    last_window = target_track[actual_len - window_size:actual_len]
                    all_windows.append(last_window)
                    all_lengths.append(window_size)
    
    if len(all_windows) == 0:
        return None, None
    
    # pad to same length
    max_len = max(w.size(0) for w in all_windows)
    num_windows = len(all_windows)
    
    padded_windows = torch.zeros(num_windows, max_len, num_points, 3, device=device, dtype=tracks.dtype)
    window_lengths = torch.zeros(num_windows, dtype=torch.long, device=device)
    
    for i, (win, wlen) in enumerate(zip(all_windows, all_lengths)):
        padded_windows[i, :win.size(0)] = win
        window_lengths[i] = wlen
    
    return padded_windows, window_lengths


# ==========================================
# Training Loop
# ==========================================

def train_mae(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value
    
    # Setup distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend='nccl', init_method='env://', 
                          world_size=WORLD_SIZE, rank=RANK)
    
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if RANK == 0:
        print(f"Device: {device}, World Size: {WORLD_SIZE}")
        print(f"Window Size: {args.window_size}, Stride Range: [{args.stride_min}, {args.stride_max}]")
        print("Loading dataset...")
    
    # Dataset
    dataset = RealWorldDataset(
        path=args.data_path,
        split='train',
        num_obs=1,
        num_action=args.num_action,
        num_history=args.num_history,
        voxel_size=args.voxel_size,
        aug=args.aug,
        aug_jitter=False,
        with_cloud=False,
        vis=False,
        num_targets=args.num_targets
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // WORLD_SIZE,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=True
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
    
    decoder_config = {
        'embed_dim': common_embed_dim,
        'patch_size': 4,
        'num_points': args.num_points,
        'output_dim': 3,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 1024,
        'dropout': 0.1
    }
    
    model = TrackEncoderMAE(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy
    ).to(device)
    
    if RANK == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {n_params/1e6:.2f}M")
    
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False
    )
    
    if args.resume_ckpt is not None:
        model.module.load_state_dict(torch.load(args.resume_ckpt, map_location=device))
        if RANK == 0:
            print(f"Resumed from: {args.resume_ckpt}")
            
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=[0.9, 0.95], weight_decay=0.05)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * args.num_epochs
    )
    
    if args.resume_epoch >= 0:
        lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1
    
    # Training loop
    train_history = []
    model.train()
    
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0:
            print(f"\nEpoch {epoch} Starting...")
        
        sampler.set_epoch(epoch)
        epoch_loss = 0
        num_batches = 0
        total_windows = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if RANK == 0 else dataloader
        
        for data in pbar:
            # Only load human tracks
            human_tracks = data['human_tracks_rel'].to(device)
            human_lengths = data['human_track_lengths'].to(device)
            
            batch_size = human_tracks.shape[0]
            human_T = human_tracks.shape[1]
            
            # Reshape: (B, T, num_targets*num_points, 3) -> (B, T, num_targets, num_points, 3)
            human_tracks = human_tracks.view(batch_size, human_T, args.num_targets, args.num_points, 3)
            
            # Extract sliding windows
            windows, window_lengths = extract_sliding_windows(
                tracks=human_tracks,
                lengths=human_lengths,
                window_size=args.window_size,
                stride_min=args.stride_min,
                stride_max=args.stride_max,
                num_targets=args.num_targets,
                num_points=args.num_points,
                device=device
            )
            
            if windows is None:
                continue
            
            num_windows = windows.shape[0]
            total_windows += num_windows
            
            # Process in sub-batches to avoid OOM
            sub_batch_size = args.sub_batch_size
            total_loss = 0
            num_sub_batches = 0
            
            for i in range(0, num_windows, sub_batch_size):
                end_i = min(i + sub_batch_size, num_windows)
                sub_windows = windows[i:end_i]
                sub_lengths = window_lengths[i:end_i]
                
                loss, _, _ = model(sub_windows, sub_lengths)
                total_loss += loss
                num_sub_batches += 1
            
            # Average loss over sub-batches
            avg_loss = total_loss / num_sub_batches
            
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += avg_loss.item()
            num_batches += 1
            
            if RANK == 0:
                pbar.set_postfix({
                    'loss': f'{avg_loss.item():.6f}',
                    'windows': num_windows
                })
        
        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_epoch_loss = sync_loss(avg_epoch_loss, device)
        if isinstance(avg_epoch_loss, torch.Tensor):
            avg_epoch_loss = avg_epoch_loss.item()
        train_history.append(avg_epoch_loss)
        
        if RANK == 0:
            print(f"Epoch {epoch} - Avg Loss: {avg_epoch_loss:.6f}, Total Windows: {total_windows}")
            
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(args.ckpt_dir, f"mae_human_window{args.window_size}_epoch_{epoch+1}_seed_{args.seed}.ckpt")
                )
                torch.save(
                    model.module.encoder.state_dict(),
                    os.path.join(args.ckpt_dir, f"encoder_human_window{args.window_size}_epoch_{epoch+1}_seed_{args.seed}.ckpt")
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)
    
    if RANK == 0:
        torch.save(
            model.module.encoder.state_dict(),
            os.path.join(args.ckpt_dir, f"encoder_human_window{args.window_size}_final.ckpt")
        )
        print("Training Finished!")


# Default arguments
default_args = edict({
    "data_path": "data/collect_pens",
    "aug": True,
    "num_action": 20,
    "num_history": 5,
    "voxel_size": 0.005,
    "ckpt_dir": "logs/human_track_encoder_mae",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 1e-4,
    "batch_size": 128,
    "num_epochs": 100,
    "save_epochs": 10,
    "num_workers": 16,
    "seed": 42,
    "mask_ratio": 0.5,
    "mask_strategy": "random",
    "num_targets": 3,
    "num_points": 10,
    "window_size": 16,
    "stride_min": 1,
    "stride_max": 30,
    "sub_batch_size": 64
})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Human Track Encoder with MAE (Sliding Window)')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--num_action', type=int, default=20)
    parser.add_argument('--num_history', type=int, default=5)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--mask_strategy', type=str, default='random')
    parser.add_argument('--num_targets', type=int, default=3)
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--stride_min', type=int, default=1)
    parser.add_argument('--stride_max', type=int, default=30)
    parser.add_argument('--sub_batch_size', type=int, default=64)
    
    args = parser.parse_args()
    train_mae(vars(args))