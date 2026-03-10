import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import SiglipModel, SiglipImageProcessor
from PIL import Image
import os

# 配置
model_path = os.path.join("/data/pretrained-weights", "siglip-so400m-patch14-384")
image1_path = "/data/jingjing/data/context/realdata_sampled_20251107/train/task_0103_user_0555_scene_0001_cfg_0001/cam_104122063550/color/1762506583639.png"
image2_path = "/data/jingjing/data/context/realdata_sampled_20251107/train/task_0103_user_0555_scene_0006_cfg_0001/cam_104122063550/color/1762506872765.png"

# 加载模型
print("加载 SigLIP 模型...")
model = SiglipModel.from_pretrained(model_path, torch_dtype=torch.float32)
processor = SiglipImageProcessor.from_pretrained(model_path)
vision_model = model.vision_model
print("模型加载完成！")

# 参数设置
patch_size = 14
image_size = 384
grid_size = image_size // patch_size  # 27

def denormalize_image(image_tensor):
    """反归一化 SigLIP 处理后的图像"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image = image_tensor * std + mean
    return image

def extract_features(image_path, rotate_angle=0):
    """提取图像的 SigLIP 特征
    
    Args:
        image_path: 图像路径
        rotate_angle: 旋转角度 (0, 90, 180, 270)
    """
    image = Image.open(image_path).convert('RGB')
    
    # 如果需要旋转
    if rotate_angle != 0:
        image = image.rotate(-rotate_angle, expand=True)  # PIL逆时针为正
    
    # 使用 SigLIP 的 processor 进行预处理
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']  # [1, 3, 384, 384]
    
    # 提取特征
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
        features = outputs.last_hidden_state  # [1, num_patches, hidden_dim]
    
    # 提取 patch tokens (SigLIP 没有 CLS token)
    patch_tokens = features.squeeze(0)  # [num_patches, hidden_dim]
    patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=1)  # 归一化用于相似度计算
    
    # 处理图像用于显示
    processed_image = denormalize_image(pixel_values[0])
    processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
    processed_image = np.clip(processed_image, 0, 1)
    
    return patch_tokens_norm, processed_image

# 提取三张图像的特征
print("提取图像特征...")
features1, image1 = extract_features(image1_path)
features2, image2 = extract_features(image2_path)
features3, image3 = extract_features(image2_path, rotate_angle=0)  # 旋转90度
print(f"特征形状: {features1.shape}")

# 交互式选择参考点
class PointSelector:
    def __init__(self, image, features, grid_size, patch_size):
        self.image = image
        self.features = features
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.selected_point = None
        self.selected_patch_idx = None
        
    def on_click(self, event):
        if event.inaxes and event.button == 1:  # 左键点击
            x, y = int(event.xdata), int(event.ydata)
            
            # 转换为 patch 坐标
            col = x // self.patch_size
            row = y // self.patch_size
            
            # 确保在范围内
            col = min(max(0, col), self.grid_size - 1)
            row = min(max(0, row), self.grid_size - 1)
            
            self.selected_point = (row, col)
            self.selected_patch_idx = row * self.grid_size + col
            
            print(f"\n已选择点: 像素坐标({x}, {y}) -> Patch坐标({row}, {col}), 索引: {self.selected_patch_idx}")
            plt.close()

# 显示第一张图像并选择点
print("\n请在第一张图像上点击选择参考点...")
selector = PointSelector(image1, features1, grid_size, patch_size)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image1)
ax.set_title('点击选择参考点 (Scene 0001)', fontsize=14, fontweight='bold')
ax.axis('off')

# 绘制网格辅助线
for i in range(0, image_size, patch_size*2):
    ax.axhline(y=i, color='white', alpha=0.2, linewidth=0.5)
    ax.axvline(x=i, color='white', alpha=0.2, linewidth=0.5)

cid = fig.canvas.mpl_connect('button_press_event', selector.on_click)
plt.tight_layout()
plt.show()

# 检查是否选择了点
if selector.selected_patch_idx is None:
    print("未选择点，使用默认中心点")
    center_row, center_col = grid_size // 2, grid_size // 2
    selector.selected_patch_idx = center_row * grid_size + center_col
    selector.selected_point = (center_row, center_col)

# 获取参考特征
reference_feature = features1[selector.selected_patch_idx:selector.selected_patch_idx+1]
row, col = selector.selected_point

# 计算与三张图像的相似度
print("\n计算相似度...")

# Image 2 (原始)
cosine_sim2 = torch.mm(reference_feature, features2.T)
cosine_sim2 = cosine_sim2.squeeze(0).cpu().numpy()
similarity_map2 = cosine_sim2.reshape(grid_size, grid_size)

max_idx2 = np.argmax(similarity_map2)
max_row2, max_col2 = max_idx2 // grid_size, max_idx2 % grid_size
max_similarity2 = similarity_map2[max_row2, max_col2]

# Image 3 (旋转90度)
cosine_sim3 = torch.mm(reference_feature, features3.T)
cosine_sim3 = cosine_sim3.squeeze(0).cpu().numpy()
similarity_map3 = cosine_sim3.reshape(grid_size, grid_size)

max_idx3 = np.argmax(similarity_map3)
max_row3, max_col3 = max_idx3 // grid_size, max_idx3 % grid_size
max_similarity3 = similarity_map3[max_row3, max_col3]

print(f"\nImage 2 相似度范围: [{similarity_map2.min():.3f}, {similarity_map2.max():.3f}]")
print(f"Image 2 最相似位置: Patch({max_row2}, {max_col2}), 相似度: {max_similarity2:.3f}")
print(f"\nImage 3 (旋转90°) 相似度范围: [{similarity_map3.min():.3f}, {similarity_map3.max():.3f}]")
print(f"Image 3 最相似位置: Patch({max_row3}, {max_col3}), 相似度: {max_similarity3:.3f}")

# 可视化结果 - 两行布局
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)

# 第一行：参考图像和两张目标图像
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image1)
ax1.set_title('参考图像 (Scene 0001)\n选中的参考点', fontsize=12, fontweight='bold')
ax1.axis('off')
ref_x = col * patch_size + patch_size // 2
ref_y = row * patch_size + patch_size // 2
ax1.plot(ref_x, ref_y, 'rx', markersize=20, markeredgewidth=4, label='参考点')
ax1.legend(loc='upper right', fontsize=10)

# Image 2 (原始)
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(image2)
ax2.set_title('目标图像 (Scene 0006 - 原始)\n最相似位置标记', fontsize=12, fontweight='bold')
ax2.axis('off')
match_x2 = max_col2 * patch_size + patch_size // 2
match_y2 = max_row2 * patch_size + patch_size // 2
ax2.plot(match_x2, match_y2, 'go', markersize=20, markeredgewidth=4, 
         markerfacecolor='none', label=f'最相似点 ({max_similarity2:.3f})')
ax2.legend(loc='upper right', fontsize=10)

# Image 3 (旋转90度)
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(image3)
ax3.set_title('目标图像 (Scene 0006 - 旋转90°)\n最相似位置标记', fontsize=12, fontweight='bold')
ax3.axis('off')
match_x3 = max_col3 * patch_size + patch_size // 2
match_y3 = max_row3 * patch_size + patch_size // 2
ax3.plot(match_x3, match_y3, 'mo', markersize=20, markeredgewidth=4, 
         markerfacecolor='none', label=f'最相似点 ({max_similarity3:.3f})')
ax3.legend(loc='upper right', fontsize=10)

# 第二行：相似度热图
# 参考patch位置（用于热图）
ax4 = fig.add_subplot(gs[1, 0])
ref_map = np.zeros((grid_size, grid_size))
ref_map[row, col] = 1
ax4.imshow(ref_map, cmap='Reds', vmin=0, vmax=1)
ax4.set_title(f'参考位置\nPatch({row}, {col})', fontsize=12, fontweight='bold')
ax4.plot(col, row, 'rx', markersize=15, markeredgewidth=3)
ax4.axis('off')

# Image 2 相似度热图
ax5 = fig.add_subplot(gs[1, 1])
im2 = ax5.imshow(similarity_map2, cmap='jet', vmin=0, vmax=1)
ax5.set_title(f'语义相似度地图 (原始)\n参考: Patch({row}, {col})', 
              fontsize=12, fontweight='bold')
ax5.plot(max_col2, max_row2, 'wo', markersize=15, markeredgewidth=3, 
         markerfacecolor='none')
ax5.axis('off')
plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04, label='余弦相似度')

# Image 3 相似度热图
ax6 = fig.add_subplot(gs[1, 2])
im3 = ax6.imshow(similarity_map3, cmap='jet', vmin=0, vmax=1)
ax6.set_title(f'语义相似度地图 (旋转90°)\n参考: Patch({row}, {col})', 
              fontsize=12, fontweight='bold')
ax6.plot(max_col3, max_row3, 'wo', markersize=15, markeredgewidth=3, 
         markerfacecolor='none')
ax6.axis('off')
plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04, label='余弦相似度')

plt.suptitle('SigLIP跨图像语义相似度匹配（含旋转变换）', fontsize=16, fontweight='bold', y=0.98)

# 保存结果
output_path = 'siglip_cross_image_similarity_with_rotation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n结果已保存到: {output_path}")