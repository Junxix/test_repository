import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy
from policy.tokenizer import Sparse3DEncoder, SparsePositionalEncoding
from policy.track.model import TrackEncoder 
from utils.constants import TRACK_MIN, TRACK_MAX

class SemanticCrossAttentionMatcher(nn.Module):
    def __init__(self, hidden_dim, semantic_dim=1152, num_heads=4, dropout=0.1, temperature=0.1):
        super().__init__()
        
        # temperature for semantic matching (learnable or fixed)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
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
        self._semantic_match_weights = None
        self._temporal_attn_weights = None

    def _compute_semantic_matching(self, robot_sem, human_sem):
        """
        Stage 1: compute object matching weights based on semantic similarity
        
        Args:
            robot_sem: (B, N_robot, D_sem)
            human_sem: (B, T, N_human, D_sem) or (B, N_human, D_sem)
        Returns:
            match_weights: (B, N_robot, N_human)
        """
        # aggregate human semantics over time if needed
        # print(human_sem.shape)
        # print(robot_sem.shape)
        if human_sem.dim() == 4:
            # (B, T, N_human, D) -> (B, N_human, D) via mean pooling
            human_sem_agg = human_sem.mean(dim=1)
        else:
            human_sem_agg = human_sem
        
        # L2 normalize for cosine similarity (directly on original features)
        robot_sem_norm = F.normalize(robot_sem, dim=-1)
        human_sem_norm = F.normalize(human_sem_agg, dim=-1)
        
        # cosine similarity: (B, N_robot, N_human)
        similarity = torch.bmm(robot_sem_norm, human_sem_norm.transpose(1, 2))
        
        # soft matching with temperature
        match_weights = F.softmax(similarity / self.temperature, dim=-1)
        
        return match_weights, similarity

    def _reindex_by_matching(self, human_k, human_v, match_weights):
        """
        Reindex human K and V based on semantic matching weights
        
        Args:
            human_k: (B, T, N_human, H)
            human_v: (B, T, N_human, H)
            match_weights: (B, N_robot, N_human)
        Returns:
            matched_k: (B, T, N_robot, H)
            matched_v: (B, T, N_robot, H)
        """
        matched_k = torch.einsum('btnh, brn -> btrh', human_k, match_weights)
        matched_v = torch.einsum('btnh, brn -> btrh', human_v, match_weights)
        
        return matched_k, matched_v

    def _temporal_attention(self, robot_query, matched_k, matched_v, human_mask=None):
        """
        Stage 2: temporal cross-attention for each robot target
        
        Args:
            robot_query: (B, N_robot, H)
            matched_k: (B, T, N_robot, H)
            matched_v: (B, T, N_robot, H)
            human_mask: (B, T) - True for padded positions
        Returns:
            output: (B, N_robot, H)
        """
        B, N_robot, H = robot_query.shape
        T = matched_k.shape[1]
        
        outputs = []
        all_attn_weights = []
        
        for r in range(N_robot):
            q = robot_query[:, r:r+1, :]  # (B, 1, H)
            k = matched_k[:, :, r, :]      # (B, T, H)
            v = matched_v[:, :, r, :]      # (B, T, H)
            
            out, attn_weights = self.temporal_cross_attention(
                query=q,
                key=k,
                value=v,
                key_padding_mask=human_mask,
                need_weights=True,
                average_attn_weights=True
            )
            
            outputs.append(out)
            all_attn_weights.append(attn_weights)
        
        output = torch.cat(outputs, dim=1)  # (B, N_robot, H)
        
        if not self.training:
            self._temporal_attn_weights = torch.cat(all_attn_weights, dim=1)
        
        return output

    def _visualize_attention(self, save_path='attn_vis.png'):
        """Visualize both semantic matching and temporal attention"""
        import matplotlib.pyplot as plt
        
        if self._semantic_match_weights is None:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        sem_weights = self._semantic_match_weights[0].detach().cpu().numpy()
        ax = axes[0]
        im = ax.imshow(sem_weights, aspect='auto', cmap='Blues')
        ax.set_xlabel('Human Target')
        ax.set_ylabel('Robot Target')
        ax.set_title('Semantic Matching Weights')
        plt.colorbar(im, ax=ax)
        
        if self._temporal_attn_weights is not None:
            temp_weights = self._temporal_attn_weights[0].detach().cpu().numpy()
            ax = axes[1]
            im = ax.imshow(temp_weights, aspect='auto', cmap='Reds')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Robot Target')
            ax.set_title('Temporal Attention Weights')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def forward(self, robot_query, robot_sem, human_k, human_v, 
                human_sem, human_mask=None):
        """
        Args:
            robot_query: (B, N_robot, H) - robot track embeddings
            robot_sem: (B, N_robot, D_sem) - robot semantic features
            human_k: (B, T, N_human, H) - human history track embeddings
            human_v: (B, T, N_human, H) - human future track embeddings
            human_sem: (B, T, N_human, D_sem) - human semantic features
            human_mask: (B, T) - padding mask (True for padded)
        Returns:
            output: (B, N_robot, H)
        """
        # Stage 1: Semantic Matching
        match_weights, similarity = self._compute_semantic_matching(robot_sem, human_sem)
        
        if not self.training:
            self._semantic_match_weights = match_weights
        
        # Reindex K and V
        matched_k, matched_v = self._reindex_by_matching(human_k, human_v, match_weights)
        
        # Stage 2: Temporal Cross-Attention
        attended = self._temporal_attention(robot_query, matched_k, matched_v, human_mask)
        
        #FFN
        output = self.norm(attended)
        output = self.ffn(output)
        if not self.training:
            self._visualize_attention()
        
        return output


class HistRISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        num_history = 5,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        track_config=None,
        num_targets=2,
        num_points=10,
        track_encoder_ckpt=None,       # new
        value_encoder_ckpt=None,       # new
        value_seq_len=16               # new
    ):
        super().__init__()
        num_obs = 1
        self.num_history = num_history
        self.num_targets = num_targets
        self.num_points = num_points
        
        # Point cloud encoder
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)

        # Track encoder
        self.human_track_encoder = TrackEncoder(**track_config)
        # track_encoder_ckpt = "/data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/encoder_only_epoch_100_seed_42.ckpt"
        if track_encoder_ckpt is not None:
            print(f"[HistRISE] Loading pretrained track encoder from {track_encoder_ckpt}...")
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
        track_output_dim = track_config['output_dim'] or track_config['query_dim']
        # Value encoder (separate checkpoint, fixed window)
        self.value_seq_len = value_seq_len
        self.human_value_encoder = TrackEncoder(**track_config)
        # value_encoder_ckpt = "/data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window16/encoder_human_window16_epoch_50_seed_42.ckpt"
        if value_encoder_ckpt is not None:
            print(f"[HistRISE] Loading pretrained value encoder from {value_encoder_ckpt}...")
            ckpt = torch.load(value_encoder_ckpt, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            state_dict = ckpt['model'] if 'model' in state_dict else state_dict
            
            encoder_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                if k.startswith('human_track_encoder.') or k.startswith('human_value_encoder.'):
                    new_key = k.split('.', 1)[1] if '.' in k else k
                    encoder_dict[new_key] = v
            if len(encoder_dict) == 0:
                encoder_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            self.human_value_encoder.load_state_dict(encoder_dict, strict=True)
            for param in self.human_value_encoder.parameters():
                param.requires_grad = False
            self.human_value_encoder.eval()

        self.human_value_fusion = nn.Linear(track_output_dim, hidden_dim)
        # Track token fusion layers
        self.human_track_fusion = nn.Linear(track_output_dim, hidden_dim)
        
        # 2. Semantic Matcher
        # self.semantic_proj = nn.Linear(1152, hidden_dim)
        semantic_dim = 1152
        self.track_enc_dim = 512
        self.cross_attention_matcher = SemanticCrossAttentionMatcher(
            hidden_dim=self.track_enc_dim,
            semantic_dim=semantic_dim,
            num_heads=4,  
            dropout=dropout
        )
        # 3. Policy Headers
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        
        # Type embedding: 0=point cloud, 1=human, 2=robot, 3=matched_embedding
        self.type_embedding = nn.Embedding(4, hidden_dim)

        self.robot_position_embedding = SparsePositionalEncoding(hidden_dim)
        
        self.register_buffer('track_min', torch.tensor(TRACK_MIN, dtype=torch.float32))
        self.register_buffer('track_max', torch.tensor(TRACK_MAX, dtype=torch.float32))
        self.voxel_size = 0.005

    def denormalize_tracks(self, normalized_tracks):
        return (normalized_tracks + 1) / 2 * (self.track_max - self.track_min) + self.track_min

    def compute_robot_track_centers(self, robot_tracks, robot_effective_len):
        """
        Args:
            robot_tracks: (batch, seq_len, num_targets*num_points, 3)
            
        Returns:
            centers: (batch, num_targets, 3)
        """
        from sklearn.cluster import DBSCAN
        
        batch_size, seq_len = robot_tracks.shape[:2]
        robot_tracks_reshaped = robot_tracks.reshape(batch_size, seq_len, self.num_targets, self.num_points, 3)
        
        robot_tracks_denorm = self.denormalize_tracks(robot_tracks_reshaped)

        target_indices = (robot_effective_len - 1).long()
        batch_indices = torch.arange(batch_size, device=robot_tracks.device)
        last_frame_tracks = robot_tracks_denorm[batch_indices, target_indices]

        centers = np.zeros((batch_size, self.num_targets, 3), dtype=np.float32)
        
        for b in range(batch_size):
            for t in range(self.num_targets):
                points = last_frame_tracks[b, t].cpu().numpy()  # (num_points, 3)
                
                clustering = DBSCAN(eps=0.05, min_samples=2).fit(points)
                labels = clustering.labels_
                
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) > 0:
                    largest_cluster_label = unique_labels[np.argmax(counts)]
                    cluster_points = points[labels == largest_cluster_label]
                    center = cluster_points.mean(axis=0)
                else:
                    center = points.mean(axis=0)
                
                centers[b, t] = center
        
        centers = torch.from_numpy(centers).to(device=robot_tracks.device, dtype=robot_tracks.dtype)
        
        return centers


    def encode_human_key_value(self, human_tracks, human_track_lengths):
        batch_size, T, total_pts, _ = human_tracks.shape
        device = human_tracks.device
        h_tracks = human_tracks.view(batch_size, T, self.num_targets, self.num_points, 3)
        
        all_key_tracks, all_value_tracks = [], []
        all_key_lens, all_value_lens = [], []

        for t in range(T):
            # Key: 0 to t (unchanged)
            k_t = h_tracks[:, :t+1].transpose(1, 2).reshape(batch_size * self.num_targets, t+1, self.num_points, 3)
            all_key_tracks.append(k_t)
            all_key_lens.append(torch.full((batch_size * self.num_targets,), t+1, device=device))

            # Value: fixed window of value_seq_len starting from t
            v_list = []
            v_len_list = []
            for b in range(batch_size):
                act_len = human_track_lengths[b].item()
                end_idx = min(t + self.value_seq_len, act_len)
                curr_v = h_tracks[b, t:end_idx] if t < act_len else h_tracks[b, act_len-1:act_len]
                v_list.append(curr_v)
                v_len_list.append(curr_v.size(0))
            
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
            flat_tracks = torch.cat(track_list, dim=0)  # all have same padded length
            tokens = self.human_value_encoder(flat_tracks, lengths=torch.cat(len_list))
            embeds = self.human_value_fusion(tokens.view(tokens.size(0), -1, tokens.size(-1))).mean(dim=1)
            return embeds.view(T, batch_size, self.num_targets, -1).permute(1, 0, 2, 3)

        return encode_keys(all_key_tracks, all_key_lens), encode_values(all_value_tracks, all_value_lens)


    def encode_robot_query(self, robot_tracks, robot_track_lengths):
        tokens = self.human_track_encoder(robot_tracks, lengths=robot_track_lengths)
        batch_size, num_pts_enc, num_q, dim = tokens.shape
        tokens = self.human_track_fusion(tokens.view(batch_size, -1, dim))
        #（B, num_targets, num_points, D) -> (B, num_targets, D)
        return tokens.reshape(batch_size, self.num_targets, self.num_points, -1).mean(dim=2)


    def compute_dense_human_embeddings(self, human_tracks, human_track_lengths):
        """
        Args:
            human_tracks: (batch, T, num_targets, num_points, 3)
            human_track_lengths: (batch,)
            
        Returns:
            all_embeddings: (T, num_targets, hidden_dim)
        """
        b = 0 
        T = human_tracks.shape[1]
        num_targets = self.num_targets
        num_points = self.num_points
        actual_len = int(human_track_lengths[b].item()) if human_track_lengths is not None else T
        
        device = human_tracks.device
        all_embeddings = []

        for t in range(actual_len):
            # shape: (future_len, num_targets, num_points, 3)
            track_segment = human_tracks[b, t:actual_len]
            future_len = track_segment.shape[0]
            encoder_input = track_segment.permute(1, 0, 2, 3) 
            lengths = torch.full((num_targets,), future_len, dtype=torch.long, device=device)
            
            # output: (num_targets, num_points, num_queries, dim)
            tokens = self.human_track_encoder(encoder_input, lengths=lengths)
            
            # (num_targets, num_points * num_queries, dim)
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
        
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        point_cloud_type_embed = self.type_embedding(
            torch.zeros(batch_size, src.size(1), dtype=torch.long, device=src.device)
        )
        pos = pos + point_cloud_type_embed


        human_k, human_v = self.encode_human_key_value(human_tracks_rel, human_track_lengths)
        
        robot_query = self.encode_robot_query(robot_tracks_rel, robot_track_lengths)
        
        # Human Mask
        T = human_k.size(1)
        range_tensor = torch.arange(T, device=src.device).unsqueeze(0).expand(batch_size, -1)
        h_mask = range_tensor >= human_track_lengths.unsqueeze(1)
        
        matched_embedding = self.cross_attention_matcher(
            robot_query=robot_query,
            robot_sem=robot_semantics,  # (batch, num_targets, semantic_dim)
            human_k=human_k,
            human_v=human_v,
            human_sem=human_semantics,  # (batch, T, num_targets, semantic_dim)
            human_mask=h_mask
        )  # (batch, num_targets, track_enc_dim)
        # print("matched_embedding.shape", matched_embedding.shape)


        if not self.training:
            step_idx = 0
            scene_id = 3
            import matplotlib.pyplot as plt
            import os
            import torch.nn.functional as F
            

            human_tracks_rel = human_tracks_rel.view(batch_size, T, self.num_targets, self.num_points, 3)
            robot_tracks_rel = robot_tracks_rel.view(batch_size, -1, self.num_targets, self.num_points, 3)
            
            print("robot_tracks_rel.shape", robot_tracks_rel.shape)
            print("human_tracks_rel.shape", human_tracks_rel.shape)
            # robot_seq_len = robot_tracks_rel.shape[1]
            # robot_lengths = torch.full((batch_size,), robot_seq_len, dtype=torch.long, device=cloud.device)
            # dense_robot_embeddings = self.compute_dense_robot_embeddings(robot_tracks_rel, robot_lengths)
            # dense_embeddings = dense_robot_embeddings
            dense_embeddings = self.compute_dense_human_embeddings(human_tracks_rel, human_track_lengths)

            # ========== 1. Robot Query vs Human Key 相似度可视化 ==========
            # robot_query shape: (batch, num_targets, hidden_dim)
            # human_k shape: (batch, T_human, num_targets, hidden_dim)
            
            robot_query_vis = robot_query[0]  # (num_targets, hidden_dim)
            human_k_vis = human_k[0]  # (T_human, num_targets, hidden_dim)
            T_human = human_k_vis.shape[0]
            
            save_dir = "vis_robot_human_similarity"
            os.makedirs(save_dir, exist_ok=True)
            
            fig, axes = plt.subplots(1, self.num_targets, figsize=(8*self.num_targets, 4))
            if self.num_targets == 1:
                axes = [axes]
            
            for target_idx in range(self.num_targets):
                robot_q = robot_query_vis[target_idx:target_idx+1, :]  # (1, hidden_dim)
                human_k_target = human_k_vis[:, target_idx, :]  # (T_human, hidden_dim)
                
                robot_q_norm = F.normalize(robot_q, p=2, dim=1)  # (1, hidden_dim)
                human_k_norm = F.normalize(human_k_target, p=2, dim=1)  # (T_human, hidden_dim)
                
                # (1, T_human)
                similarity = torch.matmul(robot_q_norm, human_k_norm.T)
                similarity = (similarity + 1) / 2  # [-1,1] -> [0,1]
                similarity = similarity.squeeze(0).detach().cpu().numpy()  # (T_human,)
                
                npy_path = os.path.join(save_dir, f'query_similarity_target_{target_idx}_scene_{scene_id:04d}.npy')
                np.save(npy_path, similarity)
                
                ax = axes[target_idx]
                ax.plot(range(T_human), similarity, 'b-', linewidth=2, label='Similarity')
                ax.fill_between(range(T_human), similarity, alpha=0.3)
                
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
            
            save_dir = "vis_target_time_matrix"
            os.makedirs(save_dir, exist_ok=True)
            npy_save_path = os.path.join(save_dir, f'dense_embeddings_scene_epoch_76_{scene_id:04d}.npy')
            np.save(npy_save_path, dense_embeddings.detach().cpu().numpy())
            print(f"[Vis] Saved embeddings to {npy_save_path}")


            all_keys = dense_embeddings.permute(1, 0, 2).contiguous()
            
            # (Num_Targets * T, Dim)
            flat_keys = all_keys.view(-1, hidden_dim)
            flat_keys_norm = F.normalize(flat_keys, p=2, dim=1)
            
            # ((Num_Targets*T) x (Num_Targets*T))
            sim_matrix = torch.mm(flat_keys_norm, flat_keys_norm.t()).detach().cpu().numpy()
            
            plt.figure(figsize=(14, 12))
            im = plt.imshow(sim_matrix, cmap='viridis', vmin=0.0, vmax=1.0)
            plt.colorbar(im, label='Cosine Similarity')
            
            for i in range(1, num_targets):
                boundary = i * vis_T - 0.5 
                plt.axhline(y=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
                plt.axvline(x=boundary, color='white', linestyle='--', linewidth=1, alpha=0.7)
            
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

        if robot_track_lengths is not None:
            robot_effective_len = robot_track_lengths.float()  # (batch,)
        else:
            robot_effective_len = torch.full((batch_size,), robot_current_len, 
                                            dtype=torch.float32, device=src.device)
        robot_centers = self.compute_robot_track_centers(robot_tracks_abs, robot_effective_len)
        robot_coords_voxel = (robot_centers / self.voxel_size).long()  # (batch, num_targets, 3)
        
        robot_pos_list = []
        for b in range(batch_size):
            batch_coords = robot_coords_voxel[b]  # (num_targets, 3)
            coords_with_batch = torch.cat([
                torch.zeros(self.num_targets, 1, dtype=torch.long, device=batch_coords.device),
                batch_coords
            ], dim=1)  # (num_targets, 4)
            
            batch_pos = self.robot_position_embedding([coords_with_batch])  # list of (num_targets, hidden_dim)
            robot_pos_list.append(batch_pos[0])
        
        robot_pos = torch.stack(robot_pos_list, dim=0)  # (batch, num_targets, hidden_dim)
        
        matched_type_embed = self.type_embedding(
            torch.full((batch_size, self.num_targets), 3, dtype=torch.long, device=src.device)
        )

        matched_pos = matched_type_embed + robot_pos
        matched_embedding = matched_embedding + robot_pos

        src = torch.cat([src, matched_embedding], dim=1)
        pos = torch.cat([pos, matched_pos], dim=1)
        
        matched_padding_mask = torch.zeros(
            (batch_size, self.num_targets),
            dtype=torch.bool, device=src.device
        )
        src_padding_mask = torch.cat([src_padding_mask, matched_padding_mask], dim=1)
        
        # Transformer + Action Decoder
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
        super().train(mode)
        self.human_track_encoder.eval()
        self.human_value_encoder.eval()
        return self