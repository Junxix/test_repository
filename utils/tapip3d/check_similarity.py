import numpy as np
import os

# 设置工作目录
data_dir = "/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train/task_0103_user_0555_scene_0021_cfg_0001_BEFORE_task_0103_user_0555_scene_0028_cfg_0001_AFTER/cam_104122063550/robot_siglip"

os.chdir(data_dir)

# 加载所有数据
target_1 = np.load("target_1.npy")
target_2 = np.load("target_2.npy")
target_3 = np.load("target_3.npy")
target_4 = np.load("target_4.npy")

print(f"target_1 shape: {target_1.shape}")
print(f"target_2 shape: {target_2.shape}")
print(f"target_3 shape: {target_3.shape}")
print(f"target_4 shape: {target_4.shape}")

# 取target_1的第一个向量
query_vector = target_4[0]  # shape: (1152,)

# 合并所有target的所有向量
all_vectors = np.vstack([target_1, target_2, target_3, target_4])  # shape: (576, 1152)

# 手动计算余弦相似度
query_norm = np.linalg.norm(query_vector)
all_norms = np.linalg.norm(all_vectors, axis=1)
dot_products = np.dot(all_vectors, query_vector)
similarities = dot_products / (query_norm * all_norms)

print(f"\n相似度数组形状: {similarities.shape}")
print(f"最大相似度: {similarities.max():.4f}")
print(f"最小相似度: {similarities.min():.4f}")
print(f"平均相似度: {similarities.mean():.4f}")

# 各个target文件的相似度统计
print("\n各个target文件的相似度:")
print(f"target_1: max={similarities[0:144].max():.4f}, min={similarities[0:144].min():.4f}, mean={similarities[0:144].mean():.4f}")
print(f"target_2: max={similarities[144:288].max():.4f}, min={similarities[144:288].min():.4f}, mean={similarities[144:288].mean():.4f}")
print(f"target_3: max={similarities[288:432].max():.4f}, min={similarities[288:432].min():.4f}, mean={similarities[288:432].mean():.4f}")
print(f"target_4: max={similarities[432:576].max():.4f}, min={similarities[432:576].min():.4f}, mean={similarities[432:576].mean():.4f}")

# 找出最相似的top-5向量
top5_indices = np.argsort(similarities)[-5:][::-1]
print("\nTop-5 最相似的向量:")
for i, idx in enumerate(top5_indices):
    if idx < 144:
        file_name = "target_1"
        in_file_idx = idx
    elif idx < 288:
        file_name = "target_2"
        in_file_idx = idx - 144
    elif idx < 432:
        file_name = "target_3"
        in_file_idx = idx - 288
    else:
        file_name = "target_4"
        in_file_idx = idx - 432
    print(f"{i+1}. {file_name}[{in_file_idx}]: {similarities[idx]:.4f}")