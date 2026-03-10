import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
from tqdm import tqdm
import glob

# 加载模型
print("加载 CLIP 模型...")
model_path = os.path.join("/data/pretrained-weights", "clip-vit-large-patch14")  # 或者使用 "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.float32)
processor = CLIPProcessor.from_pretrained(model_path)
vision_model = model.vision_model
print("模型加载完成！")

# 定义反归一化用于可视化
def denormalize_image(image_tensor):
    """反归一化CLIP处理后的图像"""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    image = image_tensor * std + mean
    return image

# 参数设置
patch_size = 14
image_size = 224  # CLIP原生是224,但可以调整到896
grid_size = image_size // patch_size  # 56
num_patches = grid_size * grid_size

# 基础路径
base_dir = "/data/jingjing/data/context/realdata_sampled_20251107/train"

# 统计信息
stats = {
    'processed': 0,
    'skipped': 0,
    'failed': 0,
    'no_color_dir': 0,
    'no_images': 0,
    'no_mask_dir': 0
}

# 循环处理 scene_0001 到 scene_0050
for scene_id in tqdm(range(25, 26), desc="处理场景"):
    scene_name = f"scene_{scene_id:04d}"
    scene_path = os.path.join(base_dir, f"task_0103_user_0555_{scene_name}_cfg_0001/cam_104122063550")
    
    # 构建路径
    color_dir = os.path.join(scene_path, "color")
    mask_dir = os.path.join(scene_path, "sam2_tapip3d_results_offline")
    output_dir = os.path.join(scene_path, "siglip_semantic_not_normalized")
    npz_path = os.path.join(output_dir, 'siglip_semantic_features_not_normalized.npz')
    
    # # 检查是否已经处理过（断点续传）
    # if os.path.exists(npz_path):
    #     print(f"{scene_name}: 已存在，跳过")
    #     stats['skipped'] += 1
    #     continue
    
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
    
    # try:
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    
    # 使用CLIP的processor进行预处理，但指定目标尺寸
    # 注意：CLIP默认是224x224，如果要用896需要手动resize
    image_resized = image.resize((image_size, image_size), Image.BICUBIC)
    inputs = processor(images=image_resized, return_tensors="pt")
    pixel_values = inputs['pixel_values']  # [1, 3, 896, 896]
    
    # 提取特征
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
        # CLIP的输出结构：
        # outputs.last_hidden_state: [1, num_patches+1, hidden_dim]
        # 第一个token是[CLS] token，后面是patch tokens
        features = outputs.last_hidden_state
    
    # 提取patch tokens (去掉[CLS] token)
    patch_tokens = features[:, 1:, :]  # [1, num_patches, hidden_dim]
    patch_tokens = patch_tokens.squeeze(0)  # [num_patches, hidden_dim]
    # patch_tokens = F.normalize(patch_tokens, p=2, dim=1)  # 如果需要归一化
    
    feature_dim = patch_tokens.shape[-1]  # CLIP-base是768
    
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
        
        # 加载 mask
        mask = Image.open(mask_path).convert('L')
        mask_resized = mask.resize((image_size, image_size), Image.NEAREST)
        mask_array = np.array(mask_resized)
        mask_binary = (mask_array > 0).astype(np.float32)
        
        # # 计算覆盖
        # selected_patches = []
        # selected_positions = []
        
        # 首先记录所有patch的coverage
        patch_coverages = []
        for row in range(grid_size):
            for col in range(grid_size):
                y_start = row * patch_size
                y_end = (row + 1) * patch_size
                x_start = col * patch_size
                x_end = (col + 1) * patch_size
                
                patch_mask = mask_binary[y_start:y_end, x_start:x_end]
                coverage = patch_mask.sum() / (patch_size * patch_size)
                patch_idx = row * grid_size + col
                patch_coverages.append({
                    'row': row,
                    'col': col,
                    'coverage': coverage,
                    'patch_idx': patch_idx
                })
                print(f"Target {target_id} Patch ({row}, {col}) coverage: {coverage:.2f}")

        # 选择patches
        selected_patches = []
        selected_positions = []

        # 找出所有coverage >= 0.5的patches
        high_coverage_patches = [p for p in patch_coverages if p['coverage'] >= 0.5]

        if len(high_coverage_patches) > 0:
            # 如果有coverage >= 0.5的patches，选择它们
            for p in high_coverage_patches:
                selected_patches.append(p['patch_idx'])
                selected_positions.append((p['row'], p['col']))
            print(f"Target {target_id}: 选择了 {len(high_coverage_patches)} 个 coverage >= 0.5 的patches")
        else:
            # 如果没有，选择coverage最大的那个
            max_coverage_patch = max(patch_coverages, key=lambda x: x['coverage'])
            if max_coverage_patch['coverage'] > 0:  # 确保至少有一点覆盖
                selected_patches.append(max_coverage_patch['patch_idx'])
                selected_positions.append((max_coverage_patch['row'], max_coverage_patch['col']))
                print(f"Target {target_id}: 所有patches的coverage都 < 0.5，选择最大coverage={max_coverage_patch['coverage']:.2f}的patch ({max_coverage_patch['row']}, {max_coverage_patch['col']})")
            else:
                print(f"Target {target_id}: mask完全为空，无有效patch")
        
        # 计算特征和可视化
        if len(selected_patches) > 0:
            selected_features = patch_tokens[selected_patches]
            reference_feature = selected_features.mean(dim=0, keepdim=True)
            # reference_feature = F.normalize(reference_feature, p=2, dim=1)
            
            print(reference_feature.shape)
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
            axes[1, 0].set_title(f'Selected Patches Grid', fontsize=12)
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[1, 1].imshow(similarity_map, cmap='jet', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Similarity Map', fontsize=12)
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            output_image_path = os.path.join(output_dir, f'clip_fully_covered_patches_similarity_target_{target_id}.png')
            plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            raise TypeError("error")
    
    # 保存特征
    np.savez(npz_path, **semantic_data)
    print(f"{scene_name}: ✓ [{image_name}] [{', '.join(target_stats)}]")
    stats['processed'] += 1
        
    # except Exception as e:
    #     print(f"{scene_name}: ✗ 错误 - {str(e)}")
    #     stats['failed'] += 1
    #     continue

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