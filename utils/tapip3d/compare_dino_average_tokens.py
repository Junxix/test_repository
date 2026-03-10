import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import glob
import seaborn as sns

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
print("加载 DINOv3 模型...")
model_path = os.path.join("/data/pretrained-weights", "dinov3-base")
dino = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
dino = dino.to(device)  # 将模型移到GPU
dino.eval()
print("模型加载完成！")

# 定义图像预处理
transform = T.Compose([
    T.Resize((896, 896), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 参数设置
patch_size = 16
grid_size = 896 // patch_size
num_patches = grid_size * grid_size
num_register_tokens = 4

# 基础路径
base_dir = "/data/jingjing/data/context/realdata_sampled_20251107/train"

# 存储每个场景的平均特征
scene_features = {}

# 只处理 scene_0001 到 scene_0008
for scene_id in tqdm(range(1, 9), desc="处理场景"):
    scene_name = f"scene_{scene_id:04d}"
    scene_path = os.path.join(base_dir, f"task_0103_user_0555_{scene_name}_cfg_0001/cam_104122063550")
    
    # 构建路径
    color_dir = os.path.join(scene_path, "color")
    
    # 检查 color 目录是否存在
    if not os.path.exists(color_dir):
        print(f"{scene_name}: color 目录不存在，跳过")
        continue
    
    # 获取 color 目录下的所有 PNG 文件
    png_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    
    if len(png_files) == 0:
        print(f"{scene_name}: color 目录下没有 PNG 图像，跳过")
        continue
    
    print(f"\n处理 {scene_name}, 共 {len(png_files)} 帧")
    
    # 存储该场景所有帧的特征
    all_frame_features = []
    
    try:
        # 处理该场景的所有帧
        for image_path in tqdm(png_files, desc=f"{scene_name} 帧", leave=False):
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 896, 896] 移到GPU
            
            # 提取特征
            with torch.no_grad():
                outputs = dino(pixel_values=image_tensor)
                features = outputs.last_hidden_state
            
            # 提取 patch tokens
            patch_tokens = features[:, 1+num_register_tokens:, :]  # [1, num_patches, dim]
            all_frame_features.append(patch_tokens)
        
        # 对所有帧的特征求平均
        all_frame_features = torch.cat(all_frame_features, dim=0)  # [num_frames, num_patches, dim]
        avg_features = all_frame_features.mean(dim=0)  # [num_patches, dim]
        
        # 保存该场景的平均特征 (保持在GPU上,后续计算更快)
        scene_features[scene_name] = avg_features
        
        print(f"{scene_name}: 成功处理 {len(png_files)} 帧, 平均特征形状: {avg_features.shape}")
        
    except Exception as e:
        print(f"{scene_name}: 处理失败 - {str(e)}")
        continue

# 打印处理结果
print(f"\n成功处理 {len(scene_features)} 个场景")
print(f"场景列表: {list(scene_features.keys())}")

# 计算场景之间的相似度
if len(scene_features) > 1:
    scene_names = sorted(scene_features.keys())
    num_scenes = len(scene_names)
    
    # 创建相似度矩阵
    similarity_matrix = np.zeros((num_scenes, num_scenes))
    
    print("\n计算场景间相似度...")
    for i in tqdm(range(num_scenes)):
        for j in range(num_scenes):
            feat_i = scene_features[scene_names[i]]  # [num_patches, dim] 在GPU上
            feat_j = scene_features[scene_names[j]]  # [num_patches, dim] 在GPU上
            
            # 方法1: 整体余弦相似度 (flatten后计算)
            feat_i_flat = feat_i.flatten()
            feat_j_flat = feat_j.flatten()
            similarity = F.cosine_similarity(feat_i_flat.unsqueeze(0), 
                                            feat_j_flat.unsqueeze(0))
            similarity_matrix[i, j] = similarity.cpu().item()  # 移到CPU
    
    # 可视化相似度矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=scene_names,
                yticklabels=scene_names,
                cmap='coolwarm',
                vmin=0.9, vmax=1.0,
                cbar_kws={'label': '余弦相似度'})
    plt.title('场景间DINOv3特征相似度矩阵 (整体相似度)')
    plt.tight_layout()
    plt.savefig('./scene_similarity_matrix_global.png', dpi=300)
    print("\n相似度矩阵已保存到: ./scene_similarity_matrix_global.png")
    
    # 方法2: 逐patch相似度,然后平均
    similarity_matrix_patch = np.zeros((num_scenes, num_scenes))
    
    print("\n计算逐patch平均相似度...")
    for i in tqdm(range(num_scenes)):
        for j in range(num_scenes):
            feat_i = scene_features[scene_names[i]]  # [num_patches, dim] 在GPU上
            feat_j = scene_features[scene_names[j]]  # [num_patches, dim] 在GPU上
            
            # 归一化
            feat_i_norm = F.normalize(feat_i, dim=1)  # [num_patches, dim]
            feat_j_norm = F.normalize(feat_j, dim=1)  # [num_patches, dim]
            
            # 计算每个patch的相似度
            patch_similarities = (feat_i_norm * feat_j_norm).sum(dim=1)  # [num_patches]
            
            # 平均
            avg_similarity = patch_similarities.mean().cpu().item()  # 移到CPU
            similarity_matrix_patch[i, j] = avg_similarity
    
    # 可视化逐patch平均相似度矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix_patch, 
                annot=True, 
                fmt='.3f',
                xticklabels=scene_names,
                yticklabels=scene_names,
                cmap='coolwarm',
                vmin=0.9, vmax=1.0,
                cbar_kws={'label': '余弦相似度'})
    plt.title('场景间DINOv3特征相似度矩阵 (逐patch平均)')
    plt.tight_layout()
    plt.savefig('./scene_similarity_matrix_patchwise.png', dpi=300)
    print("逐patch相似度矩阵已保存到: ./scene_similarity_matrix_patchwise.png")
    
    # 打印相似度统计
    print("\n=== 相似度统计 (整体方法) ===")
    for i in range(num_scenes):
        for j in range(i+1, num_scenes):
            print(f"{scene_names[i]} vs {scene_names[j]}: {similarity_matrix[i, j]:.4f}")
    
    print("\n=== 相似度统计 (逐patch平均) ===")
    for i in range(num_scenes):
        for j in range(i+1, num_scenes):
            print(f"{scene_names[i]} vs {scene_names[j]}: {similarity_matrix_patch[i, j]:.4f}")
    
    # 保存特征和相似度矩阵
    save_dict = {
        'scene_names': scene_names,
        'similarity_matrix_global': similarity_matrix,
        'similarity_matrix_patchwise': similarity_matrix_patch
    }
    
    # 保存每个场景的特征 (移到CPU并转为numpy)
    for scene_name in scene_names:
        save_dict[f'features_{scene_name}'] = scene_features[scene_name].cpu().numpy()
    
    np.savez('./scene_features_and_similarity.npz', **save_dict)
    print("\n特征和相似度矩阵已保存到: ./scene_features_and_similarity.npz")

else:
    print("\n处理的场景数量不足,无法计算相似度")

print("\n处理完成!")