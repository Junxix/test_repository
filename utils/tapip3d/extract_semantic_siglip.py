

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import SiglipModel, SiglipImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import glob

# 加载模型
print("加载 SigLIP 模型...")
model_path = os.path.join("/data/pretrained-weights", "siglip-so400m-patch14-384")
model = SiglipModel.from_pretrained(model_path, torch_dtype=torch.float32)
processor = SiglipImageProcessor.from_pretrained(model_path)
vision_model = model.vision_model
print("模型加载完成！")

# 定义反归一化用于可视化
def denormalize_image(image_tensor):
    """反归一化 SigLIP 处理后的图像"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image = image_tensor * std + mean
    return image

# 参数设置
patch_size = 14
image_size = 896  # 896x896分辨率
grid_size = image_size // patch_size  # 64 for 896
num_patches = grid_size * grid_size  # 4096 patches

# 基础路径
base_dir = "/data/jingjing/data/context/realdata_sampled_20251111/train"

# 统计信息
stats = {
    'processed': 0,
    'skipped': 0,
    'failed': 0,
    'no_color_dir': 0,
    'no_images': 0,
    'no_mask_dir': 0
}

# 循环处理 scene_0051 到 scene_0099
for scene_id in tqdm(range(51, 100), desc="处理场景"):
    scene_name = f"scene_{scene_id:04d}"
    scene_path = os.path.join(base_dir, f"task_0103_user_0555_{scene_name}_cfg_0001/cam_104122063550")
    
    # 构建路径
    color_dir = os.path.join(scene_path, "color")
    mask_dir = os.path.join(scene_path, "sam2_tapip3d_results_offline")
    output_dir = os.path.join(scene_path, "siglip_semantic_896_not_normalized")  
    npz_path = os.path.join(output_dir, 'siglip_semantic_features_896_not_normalized.npz')
    
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # 删除文件或符号链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除子目录
    # 检查 color 目录是否存在
    if not os.path.exists(color_dir):
        print(f"{scene_name}: color 目录不存在，跳过")
        stats['no_color_dir'] += 1
        continue
    
    # 获取 color 目录下的所有 PNG 文件
    png_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    
    if len(png_files) == 0:
        print(f"{scene_name}: color 目录下没有 PNG 图像，跳过")
        stats['no_images'] += 1
        continue
    
    # 取第一帧
    image_path = png_files[0]
    image_name = os.path.basename(image_path)
    
    # 检查 mask 目录是否存在
    if not os.path.exists(mask_dir):
        print(f"{scene_name}: Mask 目录不存在，跳过")
        stats['no_mask_dir'] += 1
        continue
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    

    image = Image.open(image_path).convert('RGB')
    
    # 手动resize到896x896并进行归一化
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # 归一化 (SigLIP使用的参数)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image_normalized = (image_array - mean) / std
    
    # 转换为tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
    pixel_values = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    # 提取特征
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        features = outputs.last_hidden_state
    
    print(features.shape)
    # 提取 patch tokens 并归一化
    patch_tokens = features.squeeze(0)  # [num_patches, hidden_dim]
    # patch_tokens = F.normalize(patch_tokens, p=2, dim=1)  # L2归一化
    
    feature_dim = patch_tokens.shape[-1]
    
    # 处理图像用于显示
    processed_image = denormalize_image(pixel_values[0])
    processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
    processed_image = np.clip(processed_image, 0, 1)
    
    # 存储所有目标的语义特征
    semantic_data = {}
    target_stats = []
    
    # 循环处理 mask_target_1 到 mask_target_4
    for target_id in range(1, 5):
        mask_path = os.path.join(mask_dir, f"mask_target_{target_id}.png")
        
        if not os.path.exists(mask_path):
            semantic_data[f'target_{target_id}_semantic_features'] = np.zeros((1, feature_dim), dtype=np.float32)
            target_stats.append(f"T{target_id}:无")
            continue
        
        # 加载 mask并resize到896x896
        mask = Image.open(mask_path).convert('L')
        mask_resized = mask.resize((image_size, image_size), Image.NEAREST)
        mask_array = np.array(mask_resized)
        mask_binary = (mask_array > 0).astype(np.float32)
        
        # 计算覆盖
        selected_patches = []
        selected_positions = []
        
        for row in range(grid_size):
            for col in range(grid_size):
                y_start = row * patch_size
                y_end = (row + 1) * patch_size
                x_start = col * patch_size
                x_end = (col + 1) * patch_size
                
                patch_mask = mask_binary[y_start:y_end, x_start:x_end]
                coverage = patch_mask.sum() / (patch_size * patch_size)
                
                if coverage >= 0.5:
                    patch_idx = row * grid_size + col
                    selected_patches.append(patch_idx)
                    selected_positions.append((row, col))
        
        # 计算特征和可视化
        if len(selected_patches) > 0:
            selected_features = patch_tokens[selected_patches]
            reference_feature = selected_features.mean(dim=0, keepdim=True)
            # reference_feature = F.normalize(reference_feature, p=2, dim=1)  # 再次归一化平均后的特征
            
            semantic_data[f'target_{target_id}_semantic_features'] = reference_feature.cpu().numpy()
            target_stats.append(f"T{target_id}:{len(selected_patches)}p")
            
            # 计算相似度
            cosine_sim = torch.mm(reference_feature, patch_tokens.T)
            similarity_map = cosine_sim.squeeze(0).cpu().numpy().reshape(grid_size, grid_size)
            
            selected_mask = np.zeros((grid_size, grid_size))
            for row, col in selected_positions:
                selected_mask[row, col] = 1
            
            # 可视化
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            
            axes[0, 0].imshow(processed_image)
            for row, col in selected_positions:
                y_start = row * patch_size
                x_start = col * patch_size
                rect = plt.Rectangle((x_start, y_start), patch_size, patch_size, 
                                    fill=False, edgecolor='red', linewidth=1)
                axes[0, 0].add_patch(rect)
            axes[0, 0].set_title(f'{scene_name} - Target {target_id} ({len(selected_patches)} patches)\nImage: {image_name}', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(processed_image)
            axes[0, 1].imshow(mask_binary, alpha=0.5, cmap='Reds')
            axes[0, 1].set_title(f'{scene_name} - Target {target_id}: Mask', fontsize=12)
            axes[0, 1].axis('off')
            
            im1 = axes[1, 0].imshow(selected_mask, cmap='Reds', vmin=0, vmax=1)
            axes[1, 0].set_title(f'Selected Patches Grid (896x896)', fontsize=12)
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[1, 1].imshow(similarity_map, cmap='jet', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Similarity Map', fontsize=12)
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            output_image_path = os.path.join(output_dir, f'siglip_fully_covered_patches_similarity_target_{target_id}.png')
            plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            semantic_data[f'target_{target_id}_semantic_features'] = np.zeros((1, feature_dim), dtype=np.float32)
            target_stats.append(f"T{target_id}:0p")
    
    # 保存特征
    np.savez(npz_path, **semantic_data)
    print(f"{scene_name}: ✓ [{image_name}] [{', '.join(target_stats)}]")
    stats['processed'] += 1
        

# 打印统计信息
print(f"\n{'='*80}")
print("处理完成！统计信息:")
print(f"  成功处理: {stats['processed']}")
print(f"  已存在跳过: {stats['skipped']}")
print(f"  处理失败: {stats['failed']}")
print(f"  color目录不存在: {stats['no_color_dir']}")
print(f"  没有PNG图像: {stats['no_images']}")
print(f"  Mask目录不存在: {stats['no_mask_dir']}")
print(f"{'='*80}")