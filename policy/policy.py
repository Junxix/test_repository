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

# policy/policy.py
class SemanticCrossAttentionMatcher(nn.Module):
    def __init__(self, hidden_dim, semantic_dim=1152, num_heads=4, dropout=0.1, temperature=0.1):
        super().__init__()
        
        # temporal cross-attention (stage 2)
        self.temporal_cross_attention = nn.MultiheadAttention(
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
        
        # for visualization
        self._semantic_match_indices = None
        self._similarity_matrix = None
        self._temporal_attn_weights = None

    def _compute_semantic_matching(self, robot_sem, human_sem):
        """
        Stage 1: hard matching based on semantic similarity (argmax, no learning)
        
        Args:
            robot_sem: (B, N_robot, D_sem)
            human_sem: (B, T, N_human, D_sem) or (B, N_human, D_sem)
        Returns:
            match_indices: (B, N_robot) - index of best matching human target
            similarity: (B, N_robot, N_human)
        """
        # aggregate human semantics over time if needed
        if human_sem.dim() == 4:
            human_sem_agg = human_sem.mean(dim=1)
        else:
            human_sem_agg = human_sem
        
        # print(human_sem.shape)
        # L2 normalize for cosine similarity
        robot_sem_norm = F.normalize(robot_sem, dim=-1)
        human_sem_norm = F.normalize(human_sem_agg, dim=-1)
        
        # cosine similarity: (B, N_robot, N_human)
        similarity = torch.bmm(robot_sem_norm, human_sem_norm.transpose(1, 2))
        
        # hard matching: select the most similar human target
        match_indices = similarity.argmax(dim=-1)  # (B, N_robot)
        
        return match_indices, similarity

    def _reindex_by_matching(self, human_k, human_v, match_indices):
        """
        Reindex human K and V based on hard matching indices
        
        Args:
            human_k: (B, T, N_human, H)
            human_v: (B, T, N_human, H)
            match_indices: (B, N_robot)
        Returns:
            matched_k: (B, T, N_robot, H)
            matched_v: (B, T, N_robot, H)
        """
        B, T, N_human, H = human_k.shape
        N_robot = match_indices.shape[1]
        
        # expand indices for gathering: (B, T, N_robot, H)
        idx = match_indices[:, None, :, None].expand(B, T, N_robot, H)
        
        matched_k = torch.gather(human_k, dim=2, index=idx)
        matched_v = torch.gather(human_v, dim=2, index=idx)
        
        return matched_k, matched_v

    def _temporal_attention(self, robot_query, matched_k, matched_v, human_mask=None):
        """
        Stage 2: temporal cross-attention for each robot target
        """
        B, N_robot, H = robot_query.shape
        T = matched_k.shape[1]
        
        outputs = []
        all_attn_weights = []
        
        for r in range(N_robot):
            q = robot_query[:, r:r+1, :]
            k = matched_k[:, :, r, :]
            v = matched_v[:, :, r, :]
            
            out, attn_weights = self.temporal_cross_attention(
                query=q, key=k, value=v,
                key_padding_mask=human_mask,
                need_weights=True,
                average_attn_weights=True
            )
            
            outputs.append(out)
            all_attn_weights.append(attn_weights)
        
        output = torch.cat(outputs, dim=1)
        
        if not self.training:
            self._temporal_attn_weights = torch.cat(all_attn_weights, dim=1)
        
        return output

    def _visualize_attention(self, save_path='attn_vis.png'):
        """Visualize semantic matching indices and temporal attention"""
        import matplotlib.pyplot as plt
        
        if self._similarity_matrix is None:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # show similarity matrix with hard matching marked
        sim = self._similarity_matrix[0].detach().cpu().numpy()
        indices = self._semantic_match_indices[0].detach().cpu().numpy()
        
        ax = axes[0]
        im = ax.imshow(sim, aspect='auto', cmap='Blues')
        # mark selected indices
        for r, h in enumerate(indices):
            ax.scatter(h, r, color='red', s=100, marker='x', linewidths=2)
        ax.set_xlabel('Human Target')
        ax.set_ylabel('Robot Target')
        ax.set_title('Semantic Similarity (X = selected)')
        plt.colorbar(im, ax=ax)
        
        if self._temporal_attn_weights is not None:
            # ============================================================
            # 新增修改区域：在可视化时计算并输出所需的统计量
            # self._temporal_attn_weights shape: (Batch, N_robot, T)
            # ============================================================
            # 使用 detach() 确保不影响计算图，虽然在可视化函数里通常已经是 no_grad 了
            weights_tensor = self._temporal_attn_weights.detach()
            
            # 1. 计算每个target对应的attn的和 (沿时间维度 T 求和)
            # Shape: (Batch, N_robot)
            attn_sum_batch = weights_tensor.sum(dim=-1)
            
            # 2. 计算每个target对应的attn最大值所在的位置 t (沿时间维度 T 求 argmax)
            # Shape: (Batch, N_robot)
            max_attn_t_batch = weights_tensor.argmax(dim=-1)

            # 为了输出清晰，我们只打印 Batch 中第一个样本 (Batch 0) 的统计信息
            # 这与下面绘图只取 [0] 是一致的
            B_idx = 0
            num_robot_targets = weights_tensor.shape[1]
            
            print(f"\n--- [Attn Stats] Batch {B_idx} Temporal Attention Statistics ---")
            print(f"Target ID | Max Attn at t | Sum Attn")
            print("-" * 40)
            for r in range(num_robot_targets):
                # 获取最大值出现的时间步索引
                t_idx = max_attn_t_batch[B_idx, r].item()
                # 获取该 Target 的总注意力权重
                total_attn = attn_sum_batch[B_idx, r].item()
                
                # (可选) 获取最大值本身，方便参考验证
                # max_val = weights_tensor[B_idx, r, t_idx].item()
                
                print(f"Target {r:2d} | t = {t_idx:4d}      | {total_attn:.4f}")
            print("-" * 40)
            # ============================================================

            temp_weights = self._temporal_attn_weights[0].detach().cpu().numpy()
            ax = axes[1]
            im = ax.imshow(temp_weights, aspect='auto', cmap='Reds')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Robot Target')
            ax.set_title('Temporal Attention Weights')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to: {save_path}")
        plt.close()

    def forward(self, robot_query, robot_sem, human_k, human_v, 
                human_sem, human_mask=None):
        """
        Args:
            robot_query: (B, N_robot, H)
            robot_sem: (B, N_robot, D_sem)
            human_k: (B, T, N_human, H)
            human_v: (B, T, N_human, H)
            human_sem: (B, T, N_human, D_sem)
            human_mask: (B, T)
        Returns:
            output: (B, N_robot, H)
        """
        # Stage 1: Hard Semantic Matching (no gradient)
        with torch.no_grad():
            match_indices, similarity = self._compute_semantic_matching(robot_sem, human_sem)
        
        if not self.training:
            self._semantic_match_indices = match_indices
            self._similarity_matrix = similarity
        
        # Reindex K and V by hard matching
        matched_k, matched_v = self._reindex_by_matching(human_k, human_v, match_indices)
        
        # Stage 2: Temporal Cross-Attention
        attended = self._temporal_attention(robot_query, matched_k, matched_v, human_mask)
        
        # FFN
        output = self.norm(attended)
        output = self.ffn(output)
        
        if not self.training:
            self._visualize_attention()
        
        return output

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
        num_targets = 2,
        num_points = 10,
        track_encoder_ckpt=None,       # new
        value_encoder_ckpt=None,       # new
        value_seq_len = 48 
    ):
        super().__init__()
        num_obs = 1
        self.num_targets = num_targets
        self.num_points = num_points
        self.obs_feature_dim = obs_feature_dim
        self.voxel_size = 0.005
        self.value_seq_len = value_seq_len 
        self.register_buffer('track_min', torch.tensor(TRACK_MIN, dtype=torch.float32))
        self.register_buffer('track_max', torch.tensor(TRACK_MAX, dtype=torch.float32))

        # 1. Point Cloud Encoder
        cloud_enc_dim = 128
        self.track_enc_dim = 128
        self.sparse_encoder = SparseEncoder(cloud_enc_dim=cloud_enc_dim, input_dim=input_dim)

        # 2. Track Encoder for Key (Pretrained)
        self.human_track_encoder = TrackEncoder(**track_config)
        # track_encoder_ckpt = "/data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/encoder_only_epoch_100_seed_42.ckpt"
        if track_encoder_ckpt is not None:
            print(f"[HistRISE] Loading pretrained track encoder (Key) from {track_encoder_ckpt}...")
            ckpt = torch.load(track_encoder_ckpt, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            state_dict = ckpt['model'] if 'model' in state_dict else state_dict
            
            encoder_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                if k.startswith('human_track_encoder.'):
                    new_key = k[len('human_track_encoder.'):]
                    encoder_dict[new_key] = v
            if len(encoder_dict) == 0:
                encoder_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.human_track_encoder.load_state_dict(encoder_dict, strict=True)
            for param in self.human_track_encoder.parameters():
                param.requires_grad = False
            self.human_track_encoder.eval()

        # 3. NEW: Track Encoder for Value (from different checkpoint)
        self.human_value_encoder = TrackEncoder(**track_config)
        # value_encoder_ckpt = "/data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window48/encoder_human_window48_epoch_50_seed_42.ckpt" 
        if value_encoder_ckpt is not None:
            print(f"[HistRISE] Loading pretrained value encoder from {value_encoder_ckpt}...")
            ckpt = torch.load(value_encoder_ckpt, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            state_dict = ckpt['model'] if 'model' in state_dict else state_dict
            
            encoder_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                if k.startswith('human_track_encoder.') or k.startswith('human_value_encoder.'):
                    new_key = k.split('.')[-1] if '.' in k else k
                    encoder_dict[new_key] = v
            if len(encoder_dict) == 0:
                encoder_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.human_value_encoder.load_state_dict(encoder_dict, strict=True)
            for param in self.human_value_encoder.parameters():
                param.requires_grad = False
            self.human_value_encoder.eval()

        track_output_dim = track_config['output_dim'] or track_config['query_dim']

        self.human_track_fusion = nn.Linear(track_output_dim, self.track_enc_dim)
        self.human_value_fusion = nn.Linear(track_output_dim, self.track_enc_dim) 
        
        # 4. Semantic Cross Attention Matcher
        semantic_dim = 1152
        self.cross_attention_matcher = SemanticCrossAttentionMatcher(
            hidden_dim=self.track_enc_dim,
            semantic_dim=semantic_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # 5. Spatial Aligner
        aligner_input_dim = cloud_enc_dim + self.track_enc_dim
        self.spatial_aligner = SpatialAligner(
            mlps=[aligner_input_dim, aligner_input_dim, 256],
            out_channels=obs_feature_dim, 
            interp_fn_mode='custom'
        )

        # 6. Transformer & Decoder
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def denormalize_tracks(self, normalized_tracks):
        """Denormalize: [-1, 1] -> Absolute coordinates (Meters)"""
        return (normalized_tracks + 1) / 2 * (self.track_max - self.track_min) + self.track_min        

    def encode_human_key_value(self, human_tracks, human_track_lengths):
        """
        Encode human trajectories for Key and Value separately
        - Key: uses human_track_encoder, encodes 0 to t
        - Value: uses human_value_encoder, encodes fixed 16 frames starting from t
        
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
            # Key: 0 to t trajectories (unchanged)
            k_t = h_tracks[:, :t+1].transpose(1, 2).reshape(batch_size * self.num_targets, t+1, self.num_points, 3)
            all_key_tracks.append(k_t)
            all_key_lens.append(torch.full((batch_size * self.num_targets,), t+1, device=device))

            # Value: fixed 16 frames starting from t
            v_list = []
            v_len_list = []
            for b in range(batch_size):
                act_len = human_track_lengths[b].item()
                # Extract from t to min(t+16, act_len)
                end_idx = min(t + self.value_seq_len, act_len)
                curr_v = h_tracks[b, t:end_idx]  # (actual_len, num_targets, num_points, 3)
                v_list.append(curr_v)
                v_len_list.append(curr_v.size(0))
            
            # Pad to value_seq_len
            v_padded = torch.zeros((batch_size, self.value_seq_len, self.num_targets, self.num_points, 3), device=device)
            for b, v_item in enumerate(v_list):
                v_padded[b, :v_item.size(0)] = v_item
            
            all_value_tracks.append(v_padded.transpose(1, 2).reshape(batch_size * self.num_targets, self.value_seq_len, self.num_points, 3))
            all_value_lens.append(torch.tensor(v_len_list, device=device).repeat_interleave(self.num_targets))

        # Encode Keys with human_track_encoder
        def encode_keys(track_list, len_list):
            max_s = max(x.size(1) for x in track_list)
            flat_tracks = torch.zeros((len(track_list) * batch_size * self.num_targets, max_s, self.num_points, 3), device=device)
            for i, tracks in enumerate(track_list):
                flat_tracks[i*batch_size*self.num_targets : (i+1)*batch_size*self.num_targets, :tracks.size(1)] = tracks
            
            tokens = self.human_track_encoder(flat_tracks, lengths=torch.cat(len_list))
            embeds = self.human_track_fusion(tokens.view(tokens.size(0), -1, tokens.size(-1))).mean(dim=1)
            return embeds.view(T, batch_size, self.num_targets, -1).permute(1, 0, 2, 3)

        # Encode Values with human_value_encoder
        def encode_values(track_list, len_list):
            # All values have same padded length (value_seq_len)
            flat_tracks = torch.cat(track_list, dim=0)  # (T*batch*targets, value_seq_len, points, 3)
            
            tokens = self.human_value_encoder(flat_tracks, lengths=torch.cat(len_list))
            embeds = self.human_value_fusion(tokens.view(tokens.size(0), -1, tokens.size(-1))).mean(dim=1)
            return embeds.view(T, batch_size, self.num_targets, -1).permute(1, 0, 2, 3)

        return encode_keys(all_key_tracks, all_key_lens), encode_values(all_value_tracks, all_value_lens)

    def encode_robot_query(self, robot_tracks, robot_track_lengths):
        """
        Encode robot tracks to generate query embeddings
        Args:
            robot_tracks: (batch, seq_len, num_targets, num_points, 3)
            robot_track_lengths: (batch,)
        Returns:
            query_embeddings: (batch, num_targets, track_enc_dim)
        """
        batch_size, seq_len, num_targets, num_points, _ = robot_tracks.shape
        device = robot_tracks.device
        
        robot_input = robot_tracks.transpose(1, 2).reshape(batch_size * num_targets, seq_len, num_points, 3)
        lengths = robot_track_lengths.unsqueeze(1).expand(-1, num_targets).reshape(-1)
        
        tokens = self.human_track_encoder(robot_input, lengths=lengths)
        _, num_points_enc, num_queries, dim = tokens.shape
        tokens = tokens.view(batch_size * num_targets, num_points_enc * num_queries, dim)
        tokens = self.human_track_fusion(tokens)
        tokens = tokens.mean(dim=1)
        
        query_embeddings = tokens.view(batch_size, num_targets, -1)
        return query_embeddings


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


        for t in range(actual_len):
            track_segment = human_tracks[b, t:actual_len]
            future_len = track_segment.shape[0]
            
            encoder_input = track_segment.permute(1, 0, 2, 3) 
            
            lengths = torch.full((num_targets,), future_len, dtype=torch.long, device=device)
            
            tokens = self.human_track_encoder(encoder_input, lengths=lengths)
            
            _, num_points_enc, num_queries, dim = tokens.shape
            tokens = tokens.view(num_targets, num_points_enc * num_queries, dim)
            tokens = self.human_track_fusion(tokens)
            
            # Mean pooling
            tokens = tokens.mean(dim=1) # (num_targets, hidden_dim)
            
            all_embeddings.append(tokens)
            
        # (T, num_targets, hidden_dim)
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
        
        # Step 4
        matched_embedding = self.cross_attention_matcher(
            robot_query=robot_query,
            robot_sem=robot_semantics,  # (batch, num_targets, semantic_dim)
            human_k=human_k,
            human_v=human_v,
            human_sem=human_semantics,  # (batch, T, num_targets, semantic_dim)
            human_mask=h_mask
        )  # (batch, num_targets, track_enc_dim)

        # if not self.training:
        #     step_idx = 0
        #     scene_id = 3
        #     import matplotlib.pyplot as plt
        #     import os
        #     import torch.nn.functional as F
            

        #     human_tracks_rel = human_tracks_rel.view(batch_size, T, self.num_targets, self.num_points, 3)
        #     robot_tracks_rel = robot_tracks_rel.view(batch_size, -1, self.num_targets, self.num_points, 3)
            
        #     print("robot_tracks_rel.shape", robot_tracks_rel.shape)
        #     print("human_tracks_rel.shape", human_tracks_rel.shape)
        #     # robot_seq_len = robot_tracks_rel.shape[1]
        #     # robot_lengths = torch.full((batch_size,), robot_seq_len, dtype=torch.long, device=cloud.device)
        #     # dense_robot_embeddings = self.compute_dense_robot_embeddings(robot_tracks_rel, robot_lengths)
        #     # dense_embeddings = dense_robot_embeddings
        #     dense_embeddings = self.compute_dense_human_embeddings(human_tracks_rel, human_track_lengths)


        #     print(dense_embeddings)

        #     vis_T = dense_embeddings.shape[0]
        #     num_targets = self.num_targets
        #     hidden_dim = dense_embeddings.shape[-1]
            
        #     save_dir = "vis_target_time_matrix"
        #     os.makedirs(save_dir, exist_ok=True)
        #     npy_save_path = os.path.join(save_dir, f'dense_embeddings_scene_epoch_76_{scene_id:04d}.npy')
        #     np.save(npy_save_path, dense_embeddings.detach().cpu().numpy())
        #     print(f"[Vis] Saved embeddings to {npy_save_path}")

        #     # 1. 调整维度顺序：(Num_Targets, T, Dim)
        #     # dense_embeddings 是 (T, Targets, Dim)，permute 成 (Targets, T, Dim)
        #     all_keys = dense_embeddings.permute(1, 0, 2).contiguous()
            
        #     # 3. 展平: (Num_Targets * T, Dim)
        #     flat_keys = all_keys.view(-1, hidden_dim)
            
        #     # 4. 归一化以便计算余弦相似度
        #     flat_keys_norm = F.normalize(flat_keys, p=2, dim=1)
            
        #     # 5. 计算相似度矩阵 ((Num_Targets*T) x (Num_Targets*T))
        #     sim_matrix = torch.mm(flat_keys_norm, flat_keys_norm.t()).detach().cpu().numpy()
            
        #     # 6. 绘图
        #     plt.figure(figsize=(14, 12))
        #     im = plt.imshow(sim_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
        #     plt.colorbar(im, label='Cosine Similarity')
            
        #     # 7. 画白色网格线区分不同的 Target
        #     for i in range(1, num_targets):
        #         boundary = i * vis_T - 0.5 
        #         plt.axhline(y=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
        #         plt.axvline(x=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
            
        #     # 8. 设置刻度
        #     tick_locs = []
        #     tick_labels = []
        #     tick_interval = 10 
            
        #     for i in range(num_targets):
        #         start_idx = i * vis_T
        #         for t in range(0, vis_T, tick_interval):
        #             tick_locs.append(start_idx + t)
        #             tick_labels.append(str(t))
            
        #     plt.xticks(tick_locs, tick_labels, rotation=90, fontsize=8)
        #     plt.yticks(tick_locs, tick_labels, fontsize=8)
            
        #     plt.xlabel('Time Step (Target 1 -> Target N)')
        #     plt.ylabel('Time Step (Target 1 -> Target N)')
            
        #     plt.title(f'Target-Time Joint Similarity Matrix\n(Block Size: {vis_T}x{vis_T})\nStep: {step_idx}')
            
        #     plt.tight_layout()
        #     plt.savefig(f'{save_dir}/matrix_step_{step_idx:04d}.png', dpi=150)
        #     plt.close()
        #     print(f"[Vis] Saved matrix to {save_dir}/matrix_step_{step_idx:04d}.png")
        # # ====================================================================
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

    def train(self, mode=True):
        """Override train to keep both encoders in eval mode"""
        super().train(mode)
        self.human_track_encoder.eval()
        self.human_value_encoder.eval()  # Also freeze value encoder
        return self