# policy/track/pretrain.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from policy.track.model import TrackEncoder


class TimePositionEmbedding(nn.Module):
    """Encode time position as a query"""
    def __init__(self, dim, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, time_indices):
        # time_indices: (batch,) or (batch, num_queries)
        return self.pe[time_indices]


class TrackDecoder(nn.Module):
    """Decode embedding + time query to reconstruct relative position"""
    def __init__(self, embed_dim, num_points, hidden_dim=512, num_layers=3):
        super().__init__()
        self.num_points = num_points
        
        # Time query projection
        self.time_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Cross attention: time query attends to track embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # MLP decoder
        layers = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_points * 3))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, track_embedding, time_query):
        """
        Args:
            track_embedding: (batch, num_points, num_queries, embed_dim)
            time_query: (batch, 1, embed_dim) - time position embedding
        Returns:
            pred_positions: (batch, num_points, 3)
        """
        batch_size = track_embedding.shape[0]
        num_points = track_embedding.shape[1]
        
        # Reshape track embedding: (batch, num_points * num_queries, embed_dim)
        track_flat = track_embedding.view(batch_size, -1, track_embedding.shape[-1])
        
        # Project to hidden dim
        track_flat = self.time_proj(track_flat)  # (batch, N, hidden)
        time_q = self.time_proj(time_query)       # (batch, 1, hidden)
        
        # Cross attention
        attn_out, _ = self.cross_attn(
            query=time_q,
            key=track_flat,
            value=track_flat
        )  # (batch, 1, hidden)
        
        # Decode to positions
        pred = self.mlp(attn_out.squeeze(1))  # (batch, num_points * 3)
        pred = pred.view(batch_size, num_points, 3)
        
        return pred


class TrackEncoderPretrainer(nn.Module):
    """
    Pretrain TrackEncoder with position reconstruction task.
    Given a track sequence, encode it and reconstruct positions at random time steps.
    """
    def __init__(self, track_config, num_points=10):
        super().__init__()
        self.num_points = num_points
        
        # Encoder
        self.encoder = TrackEncoder(**track_config)
        
        # Time position embedding
        embed_dim = track_config.get('output_dim') or track_config['query_dim']
        self.time_embed = TimePositionEmbedding(embed_dim)
        
        # Decoder
        self.decoder = TrackDecoder(
            embed_dim=embed_dim,
            num_points=num_points,
            hidden_dim=512,
            num_layers=3
        )
        
    def forward(self, tracks_rel, lengths, target_times=None):
        """
        Args:
            tracks_rel: (batch, seq_len, num_points, 3) - relative tracks
            lengths: (batch,) - actual sequence lengths
            target_times: (batch,) - target time indices to reconstruct (optional)
        Returns:
            loss: reconstruction loss
            pred_positions: predicted positions at target times
            gt_positions: ground truth positions at target times
        """
        batch_size, seq_len, num_points, _ = tracks_rel.shape
        device = tracks_rel.device
        
        # Random target times if not provided
        if target_times is None:
            target_times = torch.stack([
                torch.randint(0, lengths[i].item(), (1,), device=device).squeeze()
                for i in range(batch_size)
            ])
        
        # Encode full track sequence
        # (batch, num_points, num_queries, embed_dim)
        track_embedding = self.encoder(tracks_rel, lengths)
        
        # Get time query embedding
        time_query = self.time_embed(target_times).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Decode to reconstruct position at target time
        pred_positions = self.decoder(track_embedding, time_query)  # (batch, num_points, 3)
        
        # Get ground truth positions at target times
        gt_positions = torch.stack([
            tracks_rel[i, target_times[i]]
            for i in range(batch_size)
        ])  # (batch, num_points, 3)
        
        # Reconstruction loss
        loss = F.mse_loss(pred_positions, gt_positions)
        
        return loss, pred_positions, gt_positions
    
    def encode(self, tracks_rel, lengths):
        """Just encode, for inference"""
        return self.encoder(tracks_rel, lengths)