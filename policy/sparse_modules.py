# policy/sparse_modules.py

import torch
from torch import nn
import MinkowskiEngine as ME
import os
import numpy as np
from policy.minkowski.resnet import ResNet14Max, ResNet14Mini
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 添加这行引用

def save_feature_similarity_ply(coords, features, save_path, query_idx=None):
    """
    coords: (N, 3) numpy array, 坐标
    features: (N, C) numpy array, 高维特征
    save_path: 保存路径
    query_idx: (Optional) 指定选哪一个点作为参考点。如果不传，则随机选择。
    """
    N = coords.shape[0]
    C = features.shape[1]
    
    if N == 0:
        return

    # 1. 确定参考点 (Query Point)
    if query_idx is None or query_idx >= N or query_idx < 0:
        # 随机选择一个点
        query_idx = np.random.randint(0, N)
    
    query_feat = features[query_idx] # (C,)

    # 2. 计算余弦相似度 (Cosine Similarity)
    # Sim(A, B) = (A . B) / (|A| * |B|)
    
    # 计算范数 (L2 Norm)
    feat_norm = np.linalg.norm(features, axis=1, keepdims=True) # (N, 1)
    query_norm = np.linalg.norm(query_feat) # Scalar
    
    # 避免除以0
    feat_norm = np.maximum(feat_norm, 1e-6)
    query_norm = np.maximum(query_norm, 1e-6)
    
    # 归一化特征
    features_normalized = features / feat_norm
    query_normalized = query_feat / query_norm
    
    # 点积得到相似度 [-1, 1]
    similarity = np.dot(features_normalized, query_normalized) # (N,)
    
    # 3. 将相似度映射到颜色 (Heatmap)
    # 为了让可视化对比度最强，我们进行 Min-Max 归一化到 [0, 1]
    # 这样最相似的点是 1 (红色), 最不相似的点是 0 (蓝色)
    sim_min = similarity.min()
    sim_max = similarity.max()
    
    # (N,) -> [0, 1]
    sim_scaled = (similarity - sim_min) / (sim_max - sim_min + 1e-6)
    
    # 使用 matplotlib 的 colormap (例如 'jet' 或 'viridis')
    # jet: 蓝(低) -> 青 -> 黄 -> 红(高)
    cmap = plt.get_cmap('jet') 
    
    # cmap(x) 返回 (R, G, B, A) in [0, 1]
    colors_rgba = cmap(sim_scaled) 
    
    # 转为 uint8 [0, 255], 去掉 Alpha 通道
    rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)
    
    # 把参考点标记为白色 (可选，方便在图中找到它是谁)
    rgb[query_idx] = [255, 255, 255]

    # 4. 保存为 PLY
    with open(save_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        # 也可以保存相似度值本身作为 scalar，meshlab可以重新着色
        f.write("property float similarity\n") 
        f.write("end_header\n")
        
        for i in range(N):
            f.write(f"{coords[i, 0]:.4f} {coords[i, 1]:.4f} {coords[i, 2]:.4f} "
                    f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]} "
                    f"{similarity[i]:.4f}\n")
            
    print(f"[Vis] Saved similarity map (Ref Point Idx: {query_idx}) to {save_path}")

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SharedMLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(len(layers) - 1):
            self.blocks.append(MLPBlock(layers[i], layers[i+1]))

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.blocks(x)
        return x

class CustomWeightedInterpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_feats, selected_idxs):
        """
        src_feats: (C, m)
        selected_idxs: (n, k)
        """
        selected_idxs_expand = selected_idxs.unsqueeze(1).expand(-1, src_feats.size(0), -1) # (n, C, k)
        curr_src_feats_expand = src_feats.unsqueeze(0).expand(selected_idxs.size(0), -1, -1) # (n, C, m)
        selected_feats = torch.gather(curr_src_feats_expand, 2, selected_idxs_expand) # (n, C, k)

        ctx.save_for_backward(src_feats, selected_idxs, selected_idxs_expand)

        return selected_feats
    
    @staticmethod
    def backward(ctx, grad_out):
        src_feats, selected_idxs, selected_idxs_expand = ctx.saved_tensors

        grad_src_feats = torch.zeros_like(src_feats) # (C, m)
        grad_selected_idxs = torch.zeros_like(selected_idxs)
        selected_idxs_expand = selected_idxs_expand.permute(1, 0, 2).contiguous() # (C, n, k)
        grad_out = grad_out.permute(1, 0, 2).contiguous() # (C, n, k)

        for i in range(grad_out.size(2)):
            grad_src_feats.scatter_add_(1, selected_idxs_expand[:,:,i], grad_out[:,:,i])

        return grad_src_feats, grad_selected_idxs


class WeightedSpatialInterpolation(nn.Module):
    def __init__(self, interp_fn_mode = 'custom', num_points=10):
        super().__init__()
        assert interp_fn_mode in ['custom', 'naive']
        self.interp_fn_mode = interp_fn_mode
        self.num_points = num_points
        self.close_threshold = 10 # 定义"特别特别近"的阈值 (voxel单位)

    def forward(
        self, tgt, src, tgt_feats, src_feats, k=3
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, n, 3) tensor of the xyz positions of the target features (Point Cloud)
            src: (B, m, 3) tensor of the xyz positions of the source features (Robot Tracks)
            tgt_feats: (B, C1, n) tensor of the target features
            src_feats: (B, C2, m) tensor of the source features
        """
        interpolated_feats = []
        
        # 预先计算每个样本的Outlier Token和背景平均特征
        B, C2, m = src_feats.shape
        num_targets = m // self.num_points

        for i in range(tgt.size(0)):
            # 1. 计算所有距离 (复用原有逻辑)
            all_dists_full = torch.linalg.norm(tgt[i].unsqueeze(1)-src[i].unsqueeze(0), dim=2) # (n, m)
            all_dists, all_idxs = torch.sort(all_dists_full, dim=1)

            # 2. 原始加权插值 (Weighted Interpolation)
            if self.interp_fn_mode == 'naive':
                selected_idxs = all_idxs[:, :k].unsqueeze(1).expand(-1, src_feats.size(1), -1) # (n, C2, k)
                curr_src_feats = src_feats[i:i+1].expand(selected_idxs.size(0), -1, -1) 
                selected_feats = torch.gather(curr_src_feats, 2, selected_idxs) 
            else: # 'custom'
                selected_idxs = all_idxs[:, :k] 
                selected_feats = CustomWeightedInterpFn.apply(src_feats[i], selected_idxs) 

            weight = 1.0 / (all_dists[:, :k] + 1e-6)
            norm = torch.sum(weight, dim=1, keepdim=True)
            weight = weight / norm
            # weighted_feat: (n, C2) - 这是原始的加权结果
            weighted_feat = (selected_feats * weight.unsqueeze(1)).sum(dim=2) 

            # 3. 新增逻辑：处理 Outlier Token
            if num_targets > 1:
                # 3.1 找出余弦相似度最低的 Target (Outlier)
                # src_feats[i]: (C2, m) -> reshape -> (C2, num_targets, num_points)
                curr_feats_grouped = src_feats[i].view(C2, num_targets, self.num_points)
                # 计算每个 Target 的代表特征 (对 num_points 取平均) -> (C2, num_targets)
                target_reps = curr_feats_grouped.mean(dim=2)
                
                # 计算相似度矩阵 (num_targets, num_targets)
                target_reps_norm = F.normalize(target_reps, dim=0) # Normalize channel dim
                sim_matrix = torch.mm(target_reps_norm.t(), target_reps_norm)
                
                # 找出与其他 targets 相似度总和最小的那个 index
                sim_sum = sim_matrix.sum(dim=1)
                outlier_idx = torch.argmin(sim_sum).item()
                # print("outlier_idx", outlier_idx)
                
                # 3.2 计算其他 Tokens 的平均值 (Background/Context)
                # 掩码选择非 outlier 的 targets
                bg_indices = [idx for idx in range(num_targets) if idx != outlier_idx]
                bg_indices_tensor = torch.tensor(bg_indices, device=src_feats.device)
                bg_feats = target_reps.index_select(1, bg_indices_tensor) # (C2, num_targets-1)
                avg_bg_feat = bg_feats.mean(dim=1) # (C2,)
                
                # 3.3 判断距离：点云点是否在 Outlier Token 对应的轨迹点附近
                # 获取该 Outlier 对应的所有 points 的索引范围
                start_p = outlier_idx * self.num_points
                end_p = (outlier_idx + 1) * self.num_points
                
                # 从完整的距离矩阵中提取到 Outlier points 的距离 (n, num_points)
                dists_to_outlier = all_dists_full[:, start_p:end_p]
                
                # 找到到最近的 Outlier point 的距离 (n,)
                min_dist_to_outlier, _ = dists_to_outlier.min(dim=1)
                
                # 3.4 应用混合逻辑
                # 如果距离 < threshold，使用 weighted_feat (保留细节)
                # 否则，使用 avg_bg_feat (平均后的其他特征)
                is_close = min_dist_to_outlier < self.close_threshold
                
                # 扩展 avg_bg_feat 到 (n, C2)
                avg_bg_feat_expanded = avg_bg_feat.unsqueeze(0).expand(weighted_feat.size(0), -1)
                
                # 组合: (n, C2)
                final_feat = torch.where(is_close.unsqueeze(1), weighted_feat, avg_bg_feat_expanded)

                interpolated_feats.append(final_feat)
            else:
                # 只有一个 Target，直接使用原始结果
                interpolated_feats.append(weighted_feat)

        interpolated_feats = torch.stack(interpolated_feats, dim=0).permute(0, 2, 1) # (B, C2, n)
        interpolated_feats = torch.cat([interpolated_feats, tgt_feats], dim=1)  #(B, C2 + C1, n)

        return interpolated_feats

# class WeightedSpatialInterpolation(nn.Module):
#     def __init__(self,interp_fn_mode=None, num_points=10):
#         super().__init__()
#         self.num_points = num_points
#         self.close_threshold = 10  # voxel unit threshold

#     def forward(self, tgt, src, tgt_feats, src_feats, k=3) -> torch.Tensor:
#         """
#         Args:
#             tgt: (B, n, 3) target xyz (Point Cloud)
#             src: (B, m, 3) source xyz (Robot Tracks)
#             tgt_feats: (B, C1, n) target features
#             src_feats: (B, C2, m) source features (loaded, no grad needed)
#         """
#         print("tgt.shape", tgt.shape)
#         print("src.shape", src.shape)
#         print("tgt_feats.shape", tgt_feats.shape)
#         print(src_feats.shape)
#         B, C2, m = src_feats.shape
#         num_targets = m // self.num_points
#         interpolated_feats = []
        
#         for i in range(B):
#             # Calculate all distances
#             all_dists_full = torch.linalg.norm(
#                 tgt[i].unsqueeze(1) - src[i].unsqueeze(0), dim=2
#             )  # (n, m)
#             all_dists, all_idxs = torch.sort(all_dists_full, dim=1)
            
#             # Select k nearest neighbors
#             selected_idxs = all_idxs[:, :k]  # (n, k)
#             selected_idxs_expand = selected_idxs.unsqueeze(1).expand(
#                 -1, C2, -1
#             )  # (n, C2, k)
            
#             # Gather features
#             src_feats_expand = src_feats[i].unsqueeze(0).expand(
#                 selected_idxs.size(0), -1, -1
#             )  # (n, C2, m)
#             selected_feats = torch.gather(
#                 src_feats_expand, 2, selected_idxs_expand
#             )  # (n, C2, k)
            
#             # Weighted interpolation
#             weight = 1.0 / (all_dists[:, :k] + 1e-6)  # (n, k)
#             weight = weight / weight.sum(dim=1, keepdim=True)
#             weighted_feat = (selected_feats * weight.unsqueeze(1)).sum(dim=2)  # (n, C2)
            
#             # Handle outlier token if multiple targets exist
#             if num_targets > 1:
#                 # Reshape to (C2, num_targets, num_points)
#                 curr_feats_grouped = src_feats[i].view(C2, num_targets, self.num_points)
#                 target_reps = curr_feats_grouped.mean(dim=2)  # (C2, num_targets)
                
#                 # Find outlier (lowest similarity)
#                 target_reps_norm = F.normalize(target_reps, dim=0)
#                 sim_matrix = torch.mm(target_reps_norm.t(), target_reps_norm)
#                 sim_sum = sim_matrix.sum(dim=1)
#                 outlier_idx = torch.argmin(sim_sum).item()
                
#                 # Compute background average (excluding outlier)
#                 bg_mask = torch.ones(num_targets, dtype=torch.bool, device=src_feats.device)
#                 bg_mask[outlier_idx] = False
#                 avg_bg_feat = target_reps[:, bg_mask].mean(dim=1)  # (C2,)
                
#                 # Check distance to outlier points
#                 start_p = outlier_idx * self.num_points
#                 end_p = (outlier_idx + 1) * self.num_points
#                 dists_to_outlier = all_dists_full[:, start_p:end_p]
#                 min_dist_to_outlier = dists_to_outlier.min(dim=1)[0]  # (n,)
                
#                 # Blend: use weighted_feat if close, else use background
#                 is_close = min_dist_to_outlier < self.close_threshold
#                 avg_bg_feat_expanded = avg_bg_feat.unsqueeze(0).expand_as(weighted_feat)
#                 final_feat = torch.where(
#                     is_close.unsqueeze(1), weighted_feat, avg_bg_feat_expanded
#                 )
#                 interpolated_feats.append(final_feat)
#             else:
#                 interpolated_feats.append(weighted_feat)
        
#         # Stack and permute to (B, C2, n)
#         interpolated_feats = torch.stack(interpolated_feats, dim=0).permute(0, 2, 1)
#         return interpolated_feats

# class WeightedSpatialInterpolation(nn.Module):
#     def __init__(self, interp_fn_mode = 'custom'):
#         super().__init__()
#         assert interp_fn_mode in ['custom', 'naive']
#         self.interp_fn_mode = interp_fn_mode

#     def forward(
#         self, tgt, src, tgt_feats, src_feats, k=3
#     ) -> torch.Tensor:
#         """
#         Args:
#             tgt: (B, n, 3) tensor of the xyz positions of the target features
#             src: (B, m, 3) tensor of the xyz positions of the source features
#             tgt_feats: (B, C1, n) tensor of the target features
#             src_feats: (B, C2, m) tensor of the source features

#         Returns:
#             interp_features : (B, mlp[-1], n) tensor of the features of the interpolated features
#         """
#         interpolated_feats = []
#         for i in range(tgt.size(0)):
#             all_dists = torch.linalg.norm(tgt[i].unsqueeze(1)-src[i].unsqueeze(0), dim=2) # (n, m)
#             all_dists, all_idxs = torch.sort(all_dists, dim=1)

#             if self.interp_fn_mode == 'naive':
#                 selected_idxs = all_idxs[:, :k].unsqueeze(1).expand(-1, src_feats.size(1), -1) # (n, C2, k)
#                 curr_src_feats = src_feats[i:i+1].expand(selected_idxs.size(0), -1, -1) # (n, C2, m)
#                 selected_feats = torch.gather(curr_src_feats, 2, selected_idxs) # (n, C2, k)
#             else: # 'custom'
#                 selected_idxs = all_idxs[:, :k] # (n, k)
#                 selected_feats = CustomWeightedInterpFn.apply(src_feats[i], selected_idxs) # (n, C2, k)

#             weight = 1.0 / (all_dists[:, :k] + 1e-6)
#             norm = torch.sum(weight, dim=1, keepdim=True)
#             weight = weight / norm
#             selected_feats = (selected_feats * weight.unsqueeze(1)).sum(dim=2) # (n, C2)
#             interpolated_feats.append(selected_feats)

#         interpolated_feats = torch.stack(interpolated_feats, dim=0).permute(0, 2, 1) # (B, C2, n)
#         # interpolated_feats = torch.cat([interpolated_feats, tgt_feats], dim=1)  #(B, C2 + C1, n)

#         return interpolated_feats


class SparsePositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats=512, temperature=10000, max_pos=800):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]

        x_dim_t = torch.arange(x_steps, dtype=torch.float32)
        y_dim_t = torch.arange(y_steps, dtype=torch.float32)
        z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps)
        y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps)
        z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)

        pos_x_vector = xyz_embed / x_dim_t
        pos_y_vector = xyz_embed / y_dim_t
        pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1)
            coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1)
            coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]
            pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]
            pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list

class SparseEncoder(nn.Module):
    def __init__(self, cloud_enc_dim=128, input_dim=6):
        super().__init__()
        self.cloud_enc_dim = cloud_enc_dim
        self.cloud_encoder = ResNet14Mini(in_channels=input_dim, out_channels=cloud_enc_dim, conv1_kernel_size=3, strides=(1,1,1,2), dilations=(1,2,4,8), bn_momentum=0.02, init_pool="avg")

    def forward(self, sinput):
        soutput = self.cloud_encoder(sinput)
        return soutput

class SpatialAligner(nn.Module):
    def __init__(self, mlps, out_channels=512, interp_fn_mode='custom'):
        super().__init__()
        self.out_channels = out_channels
        self.interp = WeightedSpatialInterpolation(interp_fn_mode = interp_fn_mode)
        self.interp_proj = SharedMLP(mlps)
        self.conv = ResNet14Max(in_channels=mlps[-1], out_channels=out_channels, conv1_kernel_size=3, strides=(4,2,2,2), dilations=(4,1,1,1), bn_momentum=0.02, init_pool=None)
        self.position_embedding = SparsePositionalEncoding(out_channels)

    def forward(self, sinput, track_feat, track_coord, max_num_token=150):
        # sinput: SparseTensor (Point Cloud)
        # track_feat: (B, C, N) 
        # track_coord: (B, 3, N) 
        
        batch_size = track_feat.size(0)

        cloud_feat, cloud_coord = sinput.F, sinput.C
        cloud_feat_list = []
        for i in range(batch_size):
            cloud_mask_i = cloud_coord[:, 0] == i
            cloud_coord_i = cloud_coord[cloud_mask_i][:, 1:].unsqueeze(0) # (1, num_cloud, 3)
            cloud_feat_i = cloud_feat[cloud_mask_i].permute(1, 0).unsqueeze(0) # (1, C_cloud, num_cloud)
            
            track_coord_i = track_coord[i:i+1].permute(0, 2, 1) # (1, num_track, 3)
            track_feat_i = track_feat[i:i+1] # (1, C_track, num_track)
            
            cloud_feat_i = self.interp(cloud_coord_i.float(), track_coord_i.float(), cloud_feat_i, track_feat_i)
            cloud_feat_list.append(cloud_feat_i)
        track_dim = track_feat.size(1)

        cloud_feat = torch.cat(cloud_feat_list, dim=2)
        

        cloud_feat = self.interp_proj(cloud_feat)
        cloud_feat = cloud_feat.squeeze(0).permute(1, 0)

        sinput = ME.SparseTensor(cloud_feat, sinput.C)
        soutput = self.conv(sinput)

        feats_batch, coords_batch = soutput.F, soutput.C

        # convert to sparse tokens
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:,0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.out_channels], dtype=feats_batch.dtype, device=feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.out_channels], dtype=feats_batch.dtype, device=feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype=torch.bool, device=feats_batch.device)
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            num_token = min(max_num_token, len(feats))
            tokens[i,:num_token] = feats[:num_token]
            pos_emb[i,:num_token] = pos[:num_token]
            token_padding_mask[i,:num_token] = False
        return tokens, pos_emb, token_padding_mask