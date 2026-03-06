# model.py

import torch
import torch.nn as nn
from policy.track.layers import PointPatchEmbedding, CrossAttentionBlock, SelfAttentionBlock
class Encoder(nn.Module):
    def __init__(
        self,
        num_points=32,
        input_dim=2,
        patch_size=4,
        embed_dim=256,
        query_dim=512,
        num_queries=64,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        use_rope=False,
        use_time_embedding=True,
        max_seq_len=2000
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_time_embedding = use_time_embedding
        self.num_points = num_points
        self.num_queries = num_queries
        
        self.point_patch_embed = PointPatchEmbedding(
            num_points=num_points, patch_size=patch_size, in_dim=input_dim, embed_dim=embed_dim
        )
        
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))
        nn.init.xavier_uniform_(self.queries)
        
        self.cross_attention_block = CrossAttentionBlock(
            query_dim, embed_dim, num_heads, ff_dim, dropout, 
            use_rope=use_rope, use_time_embedding=use_time_embedding, max_seq_len=max_seq_len // patch_size
        )
        
        if self.num_queries == 1:
            self.linear_transform = nn.Sequential(
                nn.Linear(query_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, query_dim)
            )
            self.self_attention_blocks = None
        else:
            self.self_attention_blocks = nn.ModuleList([
                SelfAttentionBlock(query_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ])
            self.linear_transform = None
            
        self.final_norm = nn.LayerNorm(query_dim)
        
    def create_input_positions(self, num_patches):
        if not self.use_rope and not self.use_time_embedding: 
            return None
        patch_positions = torch.arange(num_patches)
        return patch_positions
        
    def forward(self, points, lengths=None):
        batch_size = points.size(0)
        
        patches, patch_lengths = self.point_patch_embed(points, lengths)
        num_patches, num_points = patches.size(1), patches.size(2)
        
        input_positions = self.create_input_positions(num_patches)
        if input_positions is not None:
            input_positions = input_positions.to(points.device)
        
        all_point_outputs = []
        
        for point_idx in range(num_points):
            point_patches = patches[:, :, point_idx, :]
            point_queries = self.queries.expand(batch_size, -1, -1)
            
            point_mask = None
            if patch_lengths is not None:
                point_mask = torch.arange(num_patches, device=points.device)[None, :] < patch_lengths[:, None]
            
            point_queries = self.cross_attention_block(
                point_queries, 
                point_patches, 
                point_mask, 
                input_positions
            )
            
            if self.num_queries == 1:
                point_queries = self.linear_transform(point_queries)
            else:
                for self_attn_block in self.self_attention_blocks:
                    point_queries = self_attn_block(point_queries)
            
            all_point_outputs.append(point_queries)
        
        # (batch_size, num_points, num_queries, query_dim)
        output = torch.stack(all_point_outputs, dim=1)
        
        return self.final_norm(output)


class TrackEncoder(nn.Module):
    def __init__(self, output_dim=None, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.num_points = kwargs.get('num_points')
        self.num_queries = kwargs.get('num_queries')
        
        if output_dim is not None:
            self.output_proj = nn.Linear(kwargs['query_dim'], output_dim)
        else:
            self.output_proj = None
            
    def forward(self, points, lengths=None):
        encoded = self.encoder(points, lengths)
        if self.output_proj is not None:
            encoded = self.output_proj(encoded)
        
        return encoded  # (batch_size, num_points, num_queries, dim)

