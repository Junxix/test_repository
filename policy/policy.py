# policy/policy.py

import torch
import torch.nn as nn
from torch.nn import functional as F

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.sparse_modules import SparseEncoder, SpatialAligner
from utils.constants import TRACK_MIN, TRACK_MAX
from policy.track.model import TrackEncoder 
import numpy as np

# class CrossAttentionMatcher(nn.Module):
#     """
#     Cross attention module to match robot query with human key-value pairs
#     """
#     def __init__(self, hidden_dim, num_heads=1, dropout=0.1):
#         super().__init__()
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim),
#             nn.Dropout(dropout)
#         )
        
#     def forward(self, robot_query, human_k, human_v, human_mask=None):
#         """
#         Args:
#             robot_query: (batch, num_targets, hidden_dim)
#             human_k: (batch, T, num_targets, hidden_dim)
#             human_v: (batch, T, num_targets, hidden_dim)
#             human_mask: (batch, T) boolean mask
#         Returns:
#             matched_output: (batch, num_targets, hidden_dim)
#         """
#         batch_size, T, num_targets, hidden_dim = human_k.shape
#         attended_list = []
    
#         # Process each target independently
#         for t_idx in range(num_targets):
#             # Extract embeddings for current target
#             q = robot_query[:, t_idx:t_idx+1, :]  # (batch, 1, hidden_dim)
#             k = human_k[:, :, t_idx, :]           # (batch, T, hidden_dim)
#             v = human_v[:, :, t_idx, :]           # (batch, T, hidden_dim)
            
#             # Cross attention
#             attended_target, _ = self.cross_attention(
#                 query=q,
#                 key=k,
#                 value=v,
#                 key_padding_mask=human_mask,
#                 need_weights=False
#             )
#             attended_list.append(attended_target)
        
#         # Concatenate results: (batch, num_targets, hidden_dim)
#         attended = torch.cat(attended_list, dim=1)
        
#         # Residual + FFN
#         output = self.norm(attended)
#         ffn_output = self.ffn(output)
        
#         return ffn_output


class CrossAttentionMatcher(nn.Module):
    """
    Cross attention module to match robot query with human key-value pairs
    """
    def __init__(self, hidden_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, robot_query, human_k, human_v, human_mask=None):
        """
        Args:
            robot_query: (batch, num_targets, hidden_dim)
            human_k: (batch, T, num_targets, hidden_dim)
            human_v: (batch, T, num_targets, hidden_dim)
            human_mask: (batch, T) boolean mask
        Returns:
            matched_output: (batch, num_targets, hidden_dim)
        """
        batch_size, T, num_targets, hidden_dim = human_k.shape
        attended_list = []
        self._last_attn_weights = []

        # ===== 可视化和分析准备 =====
        if not self.training:
            import matplotlib.pyplot as plt
            import os
            from scipy.stats import pearsonr
            
            save_dir = "vis_qk_projection_analysis"
            os.makedirs(save_dir, exist_ok=True)
            
            # ===== 正确提取投影矩阵 =====
            # PyTorch MultiheadAttention 默认使用 in_proj_weight 存储 Q、K、V
            if self.cross_attention._qkv_same_embed_dim:
                # 使用合并的 in_proj_weight (3*hidden_dim, hidden_dim)
                in_proj = self.cross_attention.in_proj_weight
                W_q = in_proj[:hidden_dim, :]      # 前 1/3
                W_k = in_proj[hidden_dim:2*hidden_dim, :]  # 中间 1/3
                W_v = in_proj[2*hidden_dim:, :]    # 后 1/3
            else:
                # 使用分开的投影矩阵
                W_q = self.cross_attention.q_proj_weight
                W_k = self.cross_attention.k_proj_weight
                W_v = self.cross_attention.v_proj_weight
            # ==================================
            
            # ===== 投影矩阵全局分析 =====
            print("\n" + "="*70)
            print("PROJECTION MATRIX GLOBAL ANALYSIS")
            print("="*70)
            
            identity = torch.eye(hidden_dim, device=W_q.device)
            
            # 检查是否接近单位矩阵
            diff_q = torch.norm(W_q - identity).item()
            diff_k = torch.norm(W_k - identity).item()
            
            print(f"W_q Frobenius distance from Identity: {diff_q:.4f}")
            print(f"W_k Frobenius distance from Identity: {diff_k:.4f}")
            
            if diff_q < 0.1 and diff_k < 0.1:
                print("✅ Projection matrices are close to identity (minimal transformation)")
            else:
                print("⚠️  Projection matrices significantly differ from identity")
                print("   → They ARE transforming the embedding space!")
            
            # 检查正交性
            ortho_q = torch.norm(W_q @ W_q.T - identity).item()
            ortho_k = torch.norm(W_k @ W_k.T - identity).item()
            
            print(f"\nOrthogonality (0=perfect):")
            print(f"  W_q: {ortho_q:.4f}")
            print(f"  W_k: {ortho_k:.4f}")
            
            # 检查对角元素
            diag_q = torch.diag(W_q).mean().item()
            diag_k = torch.diag(W_k).mean().item()
            print(f"\nMean diagonal values:")
            print(f"  W_q: {diag_q:.4f}")
            print(f"  W_k: {diag_k:.4f}")
            
            print("="*70 + "\n")
        # ================================


        # Process each target independently
        for t_idx in range(num_targets):
            # Extract embeddings for current target
            q = robot_query[:, t_idx:t_idx+1, :]  # (batch, 1, hidden_dim)
            k = human_k[:, :, t_idx, :]           # (batch, T, hidden_dim)
            v = human_v[:, :, t_idx, :]           # (batch, T, hidden_dim)
            


            # ===== 每个 Target 的投影分析可视化 =====
            if not self.training:
                # 1. 原始空间的余弦相似度
                q_norm = F.normalize(q, p=2, dim=-1)
                k_norm = F.normalize(k, p=2, dim=-1)
                cosine_sim_original = torch.matmul(q_norm, k_norm.transpose(-2, -1))
                cosine_sim_original = cosine_sim_original.squeeze(1)[0]  # (T,)
                
                # 2. 投影后的余弦相似度（模拟 attention 内部计算）
                q_projected = torch.matmul(q, W_q.T)  # (batch, 1, hidden_dim)
                k_projected = torch.matmul(k, W_k.T)  # (batch, T, hidden_dim)
                
                q_proj_norm = F.normalize(q_projected, p=2, dim=-1)
                k_proj_norm = F.normalize(k_projected, p=2, dim=-1)
                cosine_sim_projected = torch.matmul(q_proj_norm, k_proj_norm.transpose(-2, -1))
                cosine_sim_projected = cosine_sim_projected.squeeze(1)[0]  # (T,)
                
                # 3. Attention scores (在 softmax 之前)
                scale = 1.0 / np.sqrt(hidden_dim // self.cross_attention.num_heads)
                attention_scores_raw = torch.matmul(q_projected, k_projected.transpose(-2, -1)) * scale
                attention_scores_raw = attention_scores_raw.squeeze(1)[0]  # (T,)
                
                # 转换为 numpy
                cosine_sim_original_np = cosine_sim_original.detach().cpu().numpy()
                cosine_sim_projected_np = cosine_sim_projected.detach().cpu().numpy()
                attention_scores_np = attention_scores_raw.detach().cpu().numpy()
                
                # ===== 绘制 4 子图对比 =====
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                
                # 子图1: 原始余弦相似度
                ax = axes[0, 0]
                ax.plot(range(T), cosine_sim_original_np, 'b-', linewidth=2.5, label='Original Space')
                ax.fill_between(range(T), cosine_sim_original_np, alpha=0.3)
                max_idx_orig = cosine_sim_original_np.argmax()
                ax.plot(max_idx_orig, cosine_sim_original_np[max_idx_orig], 'r*', 
                       markersize=20, label=f'Peak: t={max_idx_orig}', zorder=5)
                ax.axvline(x=max_idx_orig, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f'Target {t_idx} - Original Cosine Similarity\n(Raw Embeddings)', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Human Time Step', fontsize=11)
                ax.set_ylabel('Cosine Similarity', fontsize=11)
                ax.set_ylim([-1.1, 1.1])
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # 子图2: 投影后余弦相似度
                ax = axes[0, 1]
                ax.plot(range(T), cosine_sim_projected_np, 'g-', linewidth=2.5, label='Projected Space (W_q, W_k)')
                ax.fill_between(range(T), cosine_sim_projected_np, alpha=0.3, color='green')
                max_idx_proj = cosine_sim_projected_np.argmax()
                ax.plot(max_idx_proj, cosine_sim_projected_np[max_idx_proj], 'r*', 
                       markersize=20, label=f'Peak: t={max_idx_proj}', zorder=5)
                ax.axvline(x=max_idx_proj, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f'Target {t_idx} - Projected Cosine Similarity\n(After Linear Transform)', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Human Time Step', fontsize=11)
                ax.set_ylabel('Cosine Similarity', fontsize=11)
                ax.set_ylim([-1.1, 1.1])
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.legend(fontsize=10)
                
                # 子图3: Attention Scores (softmax前)
                ax = axes[1, 0]
                ax.plot(range(T), attention_scores_np, 'orange', linewidth=2.5, 
                       label='Attention Scores (pre-softmax)')
                ax.fill_between(range(T), attention_scores_np, alpha=0.3, color='orange')
                max_idx_attn = attention_scores_np.argmax()
                ax.plot(max_idx_attn, attention_scores_np[max_idx_attn], 'r*', 
                       markersize=20, label=f'Peak: t={max_idx_attn}', zorder=5)
                ax.axvline(x=max_idx_attn, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f'Target {t_idx} - Attention Scores\n(Scaled by 1/√d)', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Human Time Step', fontsize=11)
                ax.set_ylabel('Score', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                
                # 子图4: 三者对比（归一化到 [0, 1]）
                ax = axes[1, 1]
                
                # 归一化函数
                def normalize_to_01(arr):
                    arr_min, arr_max = arr.min(), arr.max()
                    if arr_max - arr_min < 1e-8:
                        return np.zeros_like(arr)
                    return (arr - arr_min) / (arr_max - arr_min)
                
                orig_norm = normalize_to_01(cosine_sim_original_np)
                proj_norm = normalize_to_01(cosine_sim_projected_np)
                attn_norm = normalize_to_01(attention_scores_np)
                
                ax.plot(range(T), orig_norm, 'b-', linewidth=2.5, 
                       label='Original (normalized)', alpha=0.8)
                ax.plot(range(T), proj_norm, 'g--', linewidth=2.5, 
                       label='Projected (normalized)', alpha=0.8)
                ax.plot(range(T), attn_norm, 'orange', linewidth=2, 
                       label='Attention (normalized)', alpha=0.6, linestyle=':')
                
                # 标注各自的峰值
                ax.plot(max_idx_orig, orig_norm[max_idx_orig], 'bo', markersize=10)
                ax.plot(max_idx_proj, proj_norm[max_idx_proj], 'go', markersize=10)
                ax.plot(max_idx_attn, attn_norm[max_idx_attn], 'o', 
                       color='orange', markersize=10)
                
                ax.set_title(f'Target {t_idx} - Normalized Comparison\n(All scaled to [0,1])', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Human Time Step', fontsize=11)
                ax.set_ylabel('Normalized Score [0-1]', fontsize=11)
                ax.set_ylim([-0.1, 1.1])
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'projection_analysis_target_{t_idx}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # ===== 保存 numpy 数据 =====
                np.save(os.path.join(save_dir, f'original_sim_target_{t_idx}.npy'), 
                       cosine_sim_original_np)
                np.save(os.path.join(save_dir, f'projected_sim_target_{t_idx}.npy'), 
                       cosine_sim_projected_np)
                np.save(os.path.join(save_dir, f'attention_scores_target_{t_idx}.npy'), 
                       attention_scores_np)
                
                # ===== 打印详细数值分析 =====
                print(f"\n{'='*70}")
                print(f"TARGET {t_idx} - DETAILED ANALYSIS")
                print(f"{'='*70}")
                
                print(f"\n📊 ORIGINAL Cosine Similarity (Raw Embeddings):")
                print(f"   Max: {cosine_sim_original_np.max():.4f} at t={max_idx_orig}")
                print(f"   Mean: {cosine_sim_original_np.mean():.4f}")
                print(f"   Std: {cosine_sim_original_np.std():.4f}")
                
                print(f"\n📊 PROJECTED Cosine Similarity (After W_q, W_k):")
                print(f"   Max: {cosine_sim_projected_np.max():.4f} at t={max_idx_proj}")
                print(f"   Mean: {cosine_sim_projected_np.mean():.4f}")
                print(f"   Std: {cosine_sim_projected_np.std():.4f}")
                
                print(f"\n📊 ATTENTION Scores (Before Softmax):")
                print(f"   Max: {attention_scores_np.max():.4f} at t={max_idx_attn}")
                print(f"   Mean: {attention_scores_np.mean():.4f}")
                print(f"   Std: {attention_scores_np.std():.4f}")
                
                # 峰值位置偏移分析
                shift = max_idx_proj - max_idx_orig
                print(f"\n⚠️  PEAK POSITION SHIFT:")
                print(f"   Original peak:  t={max_idx_orig}")
                print(f"   Projected peak: t={max_idx_proj}")
                print(f"   Attention peak: t={max_idx_attn}")
                print(f"   Shift (Projected - Original): {shift:+d} steps")
                
                if abs(shift) > 5:
                    print(f"   🚨 SIGNIFICANT SHIFT DETECTED! Projection is changing attention focus!")
                else:
                    print(f"   ✅ Minor shift, projection has limited impact")
                
                # 计算相关系数
                try:
                    corr_orig_proj, _ = pearsonr(cosine_sim_original_np, cosine_sim_projected_np)
                    corr_orig_attn, _ = pearsonr(cosine_sim_original_np, attention_scores_np)
                    corr_proj_attn, _ = pearsonr(cosine_sim_projected_np, attention_scores_np)
                    
                    print(f"\n📈 CORRELATION ANALYSIS (Pearson):")
                    print(f"   Original ↔ Projected:  {corr_orig_proj:+.4f}")
                    print(f"   Original ↔ Attention:  {corr_orig_attn:+.4f}")
                    print(f"   Projected ↔ Attention: {corr_proj_attn:+.4f}")
                    
                    if corr_proj_attn > 0.95:
                        print(f"\n   ✅ Projected ≈ Attention (r={corr_proj_attn:.4f})")
                        print(f"      Confirms projection matrices determine attention pattern!")
                    
                    if abs(corr_orig_proj) < 0.5:
                        print(f"\n   🚨 Original and Projected have low correlation!")
                        print(f"      W_q and W_k are SIGNIFICANTLY reshaping the similarity space!")
                    
                except:
                    print(f"\n⚠️  Could not compute correlations (constant values?)")
                
                print(f"{'='*70}\n")
            # =============================
            # Cross attention
            attended_target, attn_weights = self.cross_attention(
                query=q,
                key=k,
                value=v,
                key_padding_mask=human_mask,
                need_weights=True,
                average_attn_weights=False
            )
            attended_list.append(attended_target)
            self._last_attn_weights.append(attn_weights)

        # Concatenate results: (batch, num_targets, hidden_dim)
        attended = torch.cat(attended_list, dim=1)
        
        # Residual + FFN
        output = self.norm(attended)
        ffn_output = self.ffn(output)
        
        return ffn_output


class RISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        num_attn_layers = 4,
        dim_feedforward = 2048, 
        dropout = 0.1,
        track_config=None,
        num_targets = 4,
        num_points = 10
    ):
        super().__init__()
        num_obs = 1
        self.num_targets = num_targets
        self.num_points = num_points
        self.obs_feature_dim = obs_feature_dim
        self.voxel_size = 0.005
        self.register_buffer('track_min', torch.tensor(TRACK_MIN, dtype=torch.float32))
        self.register_buffer('track_max', torch.tensor(TRACK_MAX, dtype=torch.float32))

        # 1. Point Cloud Encoder
        cloud_enc_dim = 128
        self.track_enc_dim = 128
        self.sparse_encoder = SparseEncoder(cloud_enc_dim=cloud_enc_dim, input_dim=input_dim)

        # 2. Track Encoder (Pretrained)
        self.human_track_encoder = TrackEncoder(**track_config)
        # track_encoder_ckpt = "/data/jingjing/chkpts/su2/rise/relative_track_encoder_semantic_camera_positionalencoding_correspondence/policy_epoch_76_seed_233.ckpt"
        # if track_encoder_ckpt is not None:
        #     print(f"[HistRISE] Loading pretrained track encoder from {track_encoder_ckpt}...")
        #     ckpt = torch.load(track_encoder_ckpt, map_location='cpu')
        #     state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        #     state_dict = ckpt['model'] if 'model' in state_dict else state_dict
            
        #     encoder_dict = {}
        #     for k, v in state_dict.items():
        #         if k.startswith('module.'): k = k[7:]
        #         if k.startswith('human_track_encoder.'):
        #             new_key = k[len('human_track_encoder.'):]
        #             encoder_dict[new_key] = v
        #     if len(encoder_dict) == 0:
        #         encoder_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        #     self.human_track_encoder.load_state_dict(encoder_dict, strict=False)
        #     for param in self.human_track_encoder.parameters():
        #         param.requires_grad = False
        #     self.human_track_encoder.eval()

        track_output_dim = track_config['output_dim'] or track_config['query_dim']

        self.human_track_fusion = nn.Linear(track_output_dim, self.track_enc_dim)
        
        # 3. Cross Attention Matcher
        self.cross_attention_matcher = CrossAttentionMatcher(
            hidden_dim=self.track_enc_dim,
            num_heads=1,
            dropout=dropout
        )
        
        # 4. Spatial Aligner
        aligner_input_dim = cloud_enc_dim + self.track_enc_dim
        self.spatial_aligner = SpatialAligner(
            mlps=[aligner_input_dim, aligner_input_dim, 256],
            out_channels=obs_feature_dim, 
            interp_fn_mode='custom'
        )

        # 5. Transformer & Decoder
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def denormalize_tracks(self, normalized_tracks):
        """Denormalize: [-1, 1] -> Absolute coordinates (Meters)"""
        return (normalized_tracks + 1) / 2 * (self.track_max - self.track_min) + self.track_min        

    def encode_human_key_value(self, human_tracks, human_track_lengths):
        """
        Parallel encoding of human trajectories for all time steps to generate Key and Value
        Args:
            human_tracks: (batch, T, num_targets*num_points, 3)
            human_track_lengths: (batch,)
        Returns:
            human_k: (batch, T, num_targets, hidden_dim)
            human_v: (batch, T, num_targets, hidden_dim)
        """
        batch_size, T, total_pts, _ = human_tracks.shape
        device = human_tracks.device
        h_tracks = human_tracks.view(batch_size, T, self.num_targets, self.num_points, 3)
        
        all_key_tracks, all_value_tracks = [], []
        all_key_lens, all_value_lens = [], []

        for t in range(T):
            # Key: 0 to t trajectories
            k_t = h_tracks[:, :t+1].transpose(1, 2).reshape(batch_size * self.num_targets, t+1, self.num_points, 3)
            all_key_tracks.append(k_t)
            all_key_lens.append(torch.full((batch_size * self.num_targets,), t+1, device=device))

            # Value: t to end trajectories (considering effective length)
            v_list = []
            v_len_list = []
            for b in range(batch_size):
                act_len = human_track_lengths[b].item()
                curr_v = h_tracks[b, t:act_len] if t < act_len else h_tracks[b, act_len-1:act_len]
                v_list.append(curr_v)
                v_len_list.append(max(1, act_len - t))
            
            max_v = max(v_len_list)
            v_padded = torch.zeros((batch_size, max_v, self.num_targets, self.num_points, 3), device=device)
            for b, v_item in enumerate(v_list):
                v_padded[b, :v_item.size(0)] = v_item
            
            all_value_tracks.append(v_padded.transpose(1, 2).reshape(batch_size * self.num_targets, max_v, self.num_points, 3))
            all_value_lens.append(torch.tensor(v_len_list, device=device).repeat_interleave(self.num_targets))

        # Unified padding and encoding
        def batch_encode(track_list, len_list):
            max_s = max(x.size(1) for x in track_list)
            flat_tracks = torch.zeros((len(track_list) * batch_size * self.num_targets, max_s, self.num_points, 3), device=device)
            for i, tracks in enumerate(track_list):
                flat_tracks[i*batch_size*self.num_targets : (i+1)*batch_size*self.num_targets, :tracks.size(1)] = tracks
            
            tokens = self.human_track_encoder(flat_tracks, lengths=torch.cat(len_list))
            # [N*T, P, Q, D] -> [N*T, D] (Mean pooling over points and queries)
            embeds = self.human_track_fusion(tokens.view(tokens.size(0), -1, tokens.size(-1))).mean(dim=1)
            return embeds.view(T, batch_size, self.num_targets, -1).permute(1, 0, 2, 3)

        return batch_encode(all_key_tracks, all_key_lens), batch_encode(all_value_tracks, all_value_lens)

    def encode_robot_query(self, robot_tracks, robot_track_lengths):
        """
        Encode robot tracks to generate query embeddings
        Args:
            robot_tracks: (batch, seq_len, num_targets, num_points, 3)
            robot_track_lengths: (batch,) - effective length of each sequence
        Returns:
            query_embeddings: (batch, num_targets, track_enc_dim)
        """
        batch_size, seq_len, num_targets, num_points, _ = robot_tracks.shape
        device = robot_tracks.device
        
        # Reshape: (batch * num_targets, seq_len, num_points, 3)
        robot_input = robot_tracks.transpose(1, 2).reshape(batch_size * num_targets, seq_len, num_points, 3)
        
        # Expand lengths for all targets
        lengths = robot_track_lengths.unsqueeze(1).expand(-1, num_targets).reshape(-1)
        
        # Encode through track encoder
        tokens = self.human_track_encoder(robot_input, lengths=lengths)
        # tokens: (batch*num_targets, num_points, num_queries, dim)
        
        # Fusion & pooling
        _, num_points_enc, num_queries, dim = tokens.shape
        tokens = tokens.view(batch_size * num_targets, num_points_enc * num_queries, dim)
        tokens = self.human_track_fusion(tokens)
        tokens = tokens.mean(dim=1)  # (batch*num_targets, track_enc_dim)
        
        # Reshape back: (batch, num_targets, track_enc_dim)
        query_embeddings = tokens.view(batch_size, num_targets, -1)
        
        return query_embeddings

    def compute_dense_robot_embeddings(self, robot_tracks, robot_track_lengths):
        """
        专门用于可视化：计算 robot_tracks 中每一个时间步 t 的 embedding。
        即计算 0 -> t 的轨迹特征，对于 t=0...T
        
        Args:
            robot_tracks: (batch, T, num_targets, num_points, 3)
            robot_track_lengths: (batch,)
            
        Returns:
            all_embeddings: (T, num_targets, hidden_dim) - 仅返回 batch 0 的结果
        """
        # 为了可视化，我们只处理 batch 中的第一个样本
        b = 0 
        T = robot_tracks.shape[1]
        num_targets = self.num_targets
        num_points = self.num_points
        actual_len = int(robot_track_lengths[b].item()) if robot_track_lengths is not None else T
        
        device = robot_tracks.device
        all_embeddings = []

        # 逐个时间步循环处理
        for t in range(actual_len):
            # 1. 截取从 0 到 t+1 的轨迹 (包含第t帧)
            # shape: (history_len, num_targets, num_points, 3)
            track_segment = robot_tracks[b, 0:t+1]  # 关键修改：从0到t+1
            history_len = track_segment.shape[0]
            
            # 2. 构造 Encoder 输入
            # 需要 reshape 成 (num_targets, history_len, num_points, 3)
            encoder_input = track_segment.permute(1, 0, 2, 3) 
            
            # 构造 lengths: (num_targets,)，所有 target 的长度都是 history_len
            lengths = torch.full((num_targets,), history_len, dtype=torch.long, device=device)
            
            # 3. 通过 Encoder (使用同一个 encoder)
            # output: (num_targets, num_points, num_queries, dim)
            tokens = self.human_track_encoder(encoder_input, lengths=lengths)
            
            # 4. Fusion & Pooling
            # (num_targets, num_points * num_queries, dim)
            _, num_points_enc, num_queries, dim = tokens.shape
            tokens = tokens.view(num_targets, num_points_enc * num_queries, dim)
            tokens = self.human_track_fusion(tokens)  
            
            # Mean pooling
            tokens = tokens.mean(dim=1) # (num_targets, hidden_dim)
            
            all_embeddings.append(tokens)
            
        # 堆叠结果: (T, num_targets, hidden_dim)
        return torch.stack(all_embeddings, dim=0)

    def compute_dense_human_embeddings(self, human_tracks, human_track_lengths):
        """
        专门用于可视化：计算 human_tracks 中每一个时间步 t 的 embedding。
        即计算 t -> End 的轨迹特征，对于 t=0...T
        
        Args:
            human_tracks: (batch, T, num_targets, num_points, 3)
            human_track_lengths: (batch,)
            
        Returns:
            all_embeddings: (T, num_targets, hidden_dim) - 仅返回 batch 0 的结果
        """
        # 为了可视化，我们只处理 batch 中的第一个样本
        b = 0 
        T = human_tracks.shape[1]
        num_targets = self.num_targets
        num_points = self.num_points
        actual_len = int(human_track_lengths[b].item()) if human_track_lengths is not None else T
        
        device = human_tracks.device
        all_embeddings = []

        # 为了避免显存爆炸，我们逐个时间步或者小批次处理
        # 这里采用逐个时间步循环处理 (Batch=1 * Num_Targets)
        for t in range(actual_len):
            # 1. 截取从 t 到 结束 的轨迹
            # shape: (future_len, num_targets, num_points, 3)
            track_segment = human_tracks[b, t:actual_len]
            future_len = track_segment.shape[0]
            
            # 2. 构造 Encoder 输入
            # 需要 reshape 成 (num_targets, future_len, num_points, 3)
            # 因为 TrackEncoder 期望 (Batch, Seq, Points, Dim)
            encoder_input = track_segment.permute(1, 0, 2, 3) 
            
            # 构造 lengths: (num_targets,)，所有 target 的长度都是 future_len
            lengths = torch.full((num_targets,), future_len, dtype=torch.long, device=device)
            
            # 3. 通过 Encoder
            # output: (num_targets, num_points, num_queries, dim)
            tokens = self.human_track_encoder(encoder_input, lengths=lengths)
            
            # 4. Fusion & Pooling (保持和 encode_human_embedding_at_ratio 一致的逻辑)
            # (num_targets, num_points * num_queries, dim)
            _, num_points_enc, num_queries, dim = tokens.shape
            tokens = tokens.view(num_targets, num_points_enc * num_queries, dim)
            tokens = self.human_track_fusion(tokens)
            
            # Mean pooling
            tokens = tokens.mean(dim=1) # (num_targets, hidden_dim)
            
            all_embeddings.append(tokens)
            
        # 堆叠结果: (T, num_targets, hidden_dim)
        return torch.stack(all_embeddings, dim=0)

    def forward(self, cloud, actions=None, 
                human_tracks_abs=None, human_tracks_rel=None,
                robot_tracks_abs=None, robot_tracks_rel=None,
                human_track_lengths=None, robot_track_lengths=None, 
                human_semantics=None, robot_semantics=None, 
                robot_total_length=None, batch_size=24):

        T = human_tracks_abs.shape[1]   
        human_tracks_abs = human_tracks_abs.view(batch_size, T, self.num_targets, self.num_points, 3)
        human_tracks_rel = human_tracks_rel.view(batch_size, T, self.num_targets, self.num_points, 3)
        robot_tracks_abs = robot_tracks_abs.view(batch_size, -1, self.num_targets, self.num_points, 3)
        robot_tracks_rel = robot_tracks_rel.view(batch_size, -1, self.num_targets, self.num_points, 3)


        # Step 1: Encode human key-value pairs
        human_k, human_v = self.encode_human_key_value(
            human_tracks_rel.view(batch_size, T, -1, 3), 
            human_track_lengths
        )  # (batch, T, num_targets, track_enc_dim)
        
        # Step 2: Encode robot query
        robot_query = self.encode_robot_query(
            robot_tracks_rel, 
            robot_track_lengths
        )  # (batch, num_targets, track_enc_dim)
        
        # Step 3: Create human mask
        range_tensor = torch.arange(T, device=cloud.F.device).unsqueeze(0).expand(batch_size, -1)
        h_mask = range_tensor >= human_track_lengths.unsqueeze(1)  # (batch, T)
        
        # Step 4: Cross Attention - Robot query attends to Human key-value
        matched_embedding = self.cross_attention_matcher(
            robot_query=robot_query,
            human_k=human_k,
            human_v=human_v,
            human_mask=h_mask
        )  # (batch, num_targets, track_enc_dim)


        if not self.training:
            step_idx = 0
            scene_id = 3
            import matplotlib.pyplot as plt
            import os
            import torch.nn.functional as F
            
            # 计算 Batch 0 的所有时间步 Embeddings
            # returns: (Actual_T, Num_Targets, Hidden_Dim)

            human_tracks_rel = human_tracks_rel.view(batch_size, T, self.num_targets, self.num_points, 3)
            robot_tracks_rel = robot_tracks_rel.view(batch_size, -1, self.num_targets, self.num_points, 3)
            
            print(human_track_lengths)
            print("robot_tracks_rel.shape", robot_tracks_rel.shape)
            print("human_tracks_rel.shape", human_tracks_rel.shape)
            robot_seq_len = robot_tracks_rel.shape[1]
            robot_lengths = torch.full((batch_size,), robot_seq_len, dtype=torch.long, device=cloud.device)
            # dense_robot_embeddings = self.compute_dense_robot_embeddings(robot_tracks_rel, robot_lengths)
            # dense_embeddings = dense_robot_embeddings
            dense_embeddings = self.compute_dense_human_embeddings(human_tracks_rel, human_track_lengths)

            # ========== 1. Robot Query vs Human Key 相似度可视化 ==========
            # robot_query shape: (batch, num_targets, hidden_dim)
            # human_k shape: (batch, T_human, num_targets, hidden_dim)
            
            robot_query_vis = robot_query[0]  # (num_targets, hidden_dim)
            human_k_vis = human_k[0]  # (T_human, num_targets, hidden_dim)
            T_human = human_k_vis.shape[0]
            
            # 创建保存目录
            save_dir = "vis_robot_human_similarity"
            os.makedirs(save_dir, exist_ok=True)
            
            # 可视化 Robot Query vs Human Key
            fig, axes = plt.subplots(1, self.num_targets, figsize=(8*self.num_targets, 4))
            if self.num_targets == 1:
                axes = [axes]
            
            for target_idx in range(self.num_targets):
                # 提取当前target的embeddings
                robot_q = robot_query_vis[target_idx:target_idx+1, :]  # (1, hidden_dim)
                human_k_target = human_k_vis[:, target_idx, :]  # (T_human, hidden_dim)
                
                # 归一化
                robot_q_norm = F.normalize(robot_q, p=2, dim=1)  # (1, hidden_dim)
                human_k_norm = F.normalize(human_k_target, p=2, dim=1)  # (T_human, hidden_dim)
                
                # 计算相似度: (1, T_human)
                similarity = torch.matmul(robot_q_norm, human_k_norm.T)
                similarity = (similarity + 1) / 2  # [-1,1] -> [0,1]
                similarity = similarity.squeeze(0).detach().cpu().numpy()  # (T_human,)
                
                # 保存数据
                npy_path = os.path.join(save_dir, f'query_similarity_target_{target_idx}_scene_{scene_id:04d}.npy')
                np.save(npy_path, similarity)
                
                # 绘制折线图
                ax = axes[target_idx]
                ax.plot(range(T_human), similarity, 'b-', linewidth=2, label='Similarity')
                ax.fill_between(range(T_human), similarity, alpha=0.3)
                
                # 标注最大值
                max_idx = similarity.argmax()
                max_val = similarity[max_idx]
                ax.plot(max_idx, max_val, 'r*', markersize=15, 
                    label=f'Max: {max_val:.3f} at t={max_idx}')
                
                ax.set_xlabel('Human Time Step', fontsize=12)
                ax.set_ylabel('Cosine Similarity [0-1]', fontsize=12)
                ax.set_title(f'Target {target_idx+1}\nRobot Query vs Human Key', fontsize=14)
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                print(f"[Target {target_idx}] Robot Query vs Human Key:")
                print(f"  Max: {max_val:.4f} at Human_t={max_idx}")
                print(f"  Mean: {similarity.mean():.4f}")
            
            plt.tight_layout()
            fig_path = os.path.join(save_dir, f'query_similarity_scene_{scene_id:04d}_step_{step_idx:04d}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[Vis] Saved robot query similarity to {fig_path}")

            print(dense_embeddings)

            vis_T = dense_embeddings.shape[0]
            num_targets = self.num_targets
            hidden_dim = dense_embeddings.shape[-1]
            
            save_dir = "vis_target_time_matrix_robot"
            os.makedirs(save_dir, exist_ok=True)
            npy_save_path = os.path.join(save_dir, f'dense_embeddings_scene_epoch_76_{scene_id:04d}.npy')
            np.save(npy_save_path, dense_embeddings.detach().cpu().numpy())
            print(f"[Vis] Saved embeddings to {npy_save_path}")

            # 1. 调整维度顺序：(Num_Targets, T, Dim)
            # dense_embeddings 是 (T, Targets, Dim)，permute 成 (Targets, T, Dim)
            all_keys = dense_embeddings.permute(1, 0, 2).contiguous()
            
            # 3. 展平: (Num_Targets * T, Dim)
            flat_keys = all_keys.view(-1, hidden_dim)
            
            # 4. 归一化以便计算余弦相似度
            flat_keys_norm = F.normalize(flat_keys, p=2, dim=1)
            
            # 5. 计算相似度矩阵 ((Num_Targets*T) x (Num_Targets*T))
            sim_matrix = torch.mm(flat_keys_norm, flat_keys_norm.t()).detach().cpu().numpy()
            
            # 6. 绘图
            plt.figure(figsize=(14, 12))
            im = plt.imshow(sim_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
            plt.colorbar(im, label='Cosine Similarity')
            
            # 7. 画白色网格线区分不同的 Target
            for i in range(1, num_targets):
                boundary = i * vis_T - 0.5 
                plt.axhline(y=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
                plt.axvline(x=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
            
            # 8. 设置刻度
            tick_locs = []
            tick_labels = []
            tick_interval = 10 
            
            for i in range(num_targets):
                start_idx = i * vis_T
                for t in range(0, vis_T, tick_interval):
                    tick_locs.append(start_idx + t)
                    tick_labels.append(str(t))
            
            plt.xticks(tick_locs, tick_labels, rotation=90, fontsize=8)
            plt.yticks(tick_locs, tick_labels, fontsize=8)
            
            plt.xlabel('Time Step (Target 1 -> Target N)')
            plt.ylabel('Time Step (Target 1 -> Target N)')
            
            plt.title(f'Target-Time Joint Similarity Matrix\n(Block Size: {vis_T}x{vis_T})\nStep: {step_idx}')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/matrix_step_{step_idx:04d}.png', dpi=150)
            plt.close()
            print(f"[Vis] Saved matrix to {save_dir}/matrix_step_{step_idx:04d}.png")
        # ====================================================================


        # Step 5: Get robot track coordinates for spatial interpolation
        robot_tracks_meters = self.denormalize_tracks(robot_tracks_abs)
        
        if robot_track_lengths is not None:
            robot_effective_len = robot_track_lengths.float()
        else:
            robot_current_len = robot_tracks_abs.shape[1]
            robot_effective_len = torch.full((batch_size,), robot_current_len, 
                                           dtype=torch.float32, device=cloud.F.device)


        target_indices = (robot_effective_len - 1).long()
        batch_indices = torch.arange(batch_size, device=cloud.F.device)
        current_robot_tracks = robot_tracks_meters[batch_indices, target_indices]

        track_coords_meters = current_robot_tracks.view(batch_size, -1, 3)
        track_coords_voxel = track_coords_meters / self.voxel_size

        # Step 6: Expand matched_embedding to points as track_feats
        track_feats = matched_embedding.unsqueeze(2).repeat(1, 1, self.num_points, 1)
        track_feats = track_feats.view(batch_size, -1, self.track_enc_dim)

        # Transpose for aligner: (B, Dim, N)
        track_coords_voxel = track_coords_voxel.transpose(1, 2).contiguous() 
        track_feats = track_feats.transpose(1, 2).contiguous() 

        # Step 7: Spatial alignment
        cloud_feat = self.sparse_encoder(cloud)
        src, pos, src_padding_mask = self.spatial_aligner(cloud_feat, track_feats, track_coords_voxel)

        # Step 8: Transformer & action prediction
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

    # def train(self, mode=True):
    #         """
    #         重写 train 方法。
    #         mode=True 时，主模型进入训练模式，但强制保持 human_track_encoder 为 eval 模式。
    #         """
    #         super().train(mode)
    #         # 强制冻结 Encoder 的状态（关闭 Dropout, 锁定 Norm 统计量）
    #         self.human_track_encoder.eval()
    #         return self