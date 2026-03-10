import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from pathlib import Path

# 基础目录
base_dir = "/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train"

# 查找所有的BEFORE_AFTER目录
def find_before_after_dirs(base_dir):
    """查找所有包含BEFORE和AFTER的目录"""
    dirs = []
    for item in Path(base_dir).iterdir():
        if item.is_dir() and '_BEFORE_' in item.name and '_AFTER' in item.name:
            dirs.append(item)
    return sorted(dirs)

# 加载target_type映射
def load_target_type_mapping(json_file):
    """加载target_type.json并返回映射字典"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    # 将target1, target2等转换为数字索引，值为类型ID
    mapping = {}
    for key, value in data.items():
        target_idx = int(key.replace('target', ''))
        mapping[target_idx] = value
    return mapping

# 计算单个场景的before和after相似度
def compute_before_after_similarity_with_type(scene_dir):
    """根据target_type计算before和after之间的相似度"""
    cam_dir = scene_dir / "cam_104122063550"
    
    before_dir = cam_dir / "before_siglip_semantic_896_not_normalized"
    after_dir = cam_dir / "after_siglip_semantic_896_not_normalized"
    
    before_sam2_dir = cam_dir / "before_sam2_tapip3d_results_offline"
    after_sam2_dir = cam_dir / "after_sam2_tapip3d_results_offline"
    
    before_file = before_dir / "siglip_semantic_features_896_not_normalized.npz"
    after_file = after_dir / "siglip_semantic_features_896_not_normalized.npz"
    
    before_type_file = before_sam2_dir / "target_type.json"
    after_type_file = after_sam2_dir / "target_type.json"
    
    # 检查文件是否存在
    if not before_file.exists():
        return None, None, None, None, f"Before特征文件不存在: {before_file}"
    if not after_file.exists():
        return None, None, None, None, f"After特征文件不存在: {after_file}"
    if not before_type_file.exists():
        return None, None, None, None, f"Before类型文件不存在: {before_type_file}"
    if not after_type_file.exists():
        return None, None, None, None, f"After类型文件不存在: {after_type_file}"
    
    try:
        # 加载target_type映射
        before_type_map = load_target_type_mapping(before_type_file)
        after_type_map = load_target_type_mapping(after_type_file)
        
        # 加载特征数据
        before_data = np.load(before_file, allow_pickle=True)
        after_data = np.load(after_file, allow_pickle=True)
        
        # 提取before特征和类型
        before_features_dict = {}  # {type_id: feature}
        before_missing = []
        for i in range(1, 5):
            key = f'target_{i}_semantic_features'
            if i in before_type_map:
                type_id = before_type_map[i]
                if key in before_data.files:
                    before_features_dict[type_id] = before_data[key].reshape(1, -1)
                else:
                    before_missing.append(f"target_{i} (type {type_id})")
        
        # 提取after特征和类型
        after_features_dict = {}  # {type_id: feature}
        after_missing = []
        for i in range(1, 5):
            key = f'target_{i}_semantic_features'
            if i in after_type_map:
                type_id = after_type_map[i]
                if key in after_data.files:
                    after_features_dict[type_id] = after_data[key].reshape(1, -1)
                else:
                    after_missing.append(f"target_{i} (type {type_id})")
        
        # 找出共同的类型ID
        before_types = set(before_features_dict.keys())
        after_types = set(after_features_dict.keys())
        common_types = sorted(before_types & after_types)
        only_before = sorted(before_types - after_types)
        only_after = sorted(after_types - before_types)
        
        # 收集不匹配信息
        mismatch_info = {
            'before_map': before_type_map,
            'after_map': after_type_map,
            'before_missing': before_missing,
            'after_missing': after_missing,
            'only_before_types': only_before,
            'only_after_types': only_after,
            'common_types': common_types
        }
        
        if len(common_types) == 0:
            error_msg = "Before和After没有共同的目标类型\n"
            error_msg += f"  Before类型: {sorted(before_types)}\n"
            error_msg += f"  After类型: {sorted(after_types)}"
            return None, None, None, mismatch_info, error_msg
        
        # 按类型ID顺序构建特征矩阵
        before_features = []
        after_features = []
        for type_id in common_types:
            before_features.append(before_features_dict[type_id])
            after_features.append(after_features_dict[type_id])
        
        before_matrix = np.vstack(before_features)
        after_matrix = np.vstack(after_features)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(before_matrix, after_matrix)
        
        return similarity_matrix, before_type_map, after_type_map, mismatch_info, None
        
    except Exception as e:
        return None, None, None, None, f"处理出错: {e}"

# 主程序
all_scene_dirs = find_before_after_dirs(base_dir)
print(f"找到 {len(all_scene_dirs)} 个BEFORE_AFTER场景\n")

all_similarity_matrices = {}
all_type_matched_similarities = []  # 存储类型匹配的相似度
all_max_cross_type_similarities = []  # 存储跨类型的最大相似度

failed_scenes = []  # 记录失败的场景
partial_match_scenes = []  # 记录部分匹配的场景

for scene_dir in all_scene_dirs:
    scene_name = scene_dir.name
    print(f"\n{'='*80}")
    print(f"场景: {scene_name}")
    
    similarity_matrix, before_type_map, after_type_map, mismatch_info, error = compute_before_after_similarity_with_type(scene_dir)
    
    if error:
        print(f"  ❌ 错误: {error}")
        failed_scenes.append({
            'scene': scene_name,
            'error': error,
            'mismatch_info': mismatch_info
        })
        continue
    
    # 检查是否有不匹配的类型
    has_mismatch = False
    if mismatch_info['before_missing']:
        print(f"  ⚠️  Before缺失特征: {mismatch_info['before_missing']}")
        has_mismatch = True
    if mismatch_info['after_missing']:
        print(f"  ⚠️  After缺失特征: {mismatch_info['after_missing']}")
        has_mismatch = True
    if mismatch_info['only_before_types']:
        print(f"  ⚠️  只在Before中的类型: {mismatch_info['only_before_types']}")
        has_mismatch = True
    if mismatch_info['only_after_types']:
        print(f"  ⚠️  只在After中的类型: {mismatch_info['only_after_types']}")
        has_mismatch = True
    
    if has_mismatch:
        partial_match_scenes.append({
            'scene': scene_name,
            'mismatch_info': mismatch_info
        })
    
    print(f"\nBefore类型映射: {before_type_map}")
    print(f"After类型映射: {after_type_map}")
    print(f"✓ 成功匹配的类型: {mismatch_info['common_types']}")
    
    all_similarity_matrices[scene_name] = {
        'matrix': similarity_matrix,
        'common_types': mismatch_info['common_types'],
        'before_map': before_type_map,
        'after_map': after_type_map,
        'mismatch_info': mismatch_info
    }
    
    print(f"\n按类型ID匹配的相似度矩阵:")
    print(f"行/列顺序: {mismatch_info['common_types']}")
    print(similarity_matrix)
    
    # 对角线元素（相同类型之间的相似度）
    diagonal = np.diag(similarity_matrix)
    all_type_matched_similarities.extend(diagonal)
    
    print(f"\n相同类型的相似度 (对角线):")
    for i, type_id in enumerate(mismatch_info['common_types']):
        print(f"  类型 {type_id}: Before vs After = {diagonal[i]:.6f}")
    
    # 非对角线最大值（跨类型的最大相似度）
    if len(mismatch_info['common_types']) > 1:
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        off_diagonal = similarity_matrix[mask]
        max_cross_type = np.max(off_diagonal)
        all_max_cross_type_similarities.append(max_cross_type)
        
        max_idx = np.where((similarity_matrix == max_cross_type) & mask)
        row_type = mismatch_info['common_types'][max_idx[0][0]]
        col_type = mismatch_info['common_types'][max_idx[1][0]]
        
        print(f"\n跨类型最大相似度: {max_cross_type:.6f}")
        print(f"  位置: Before类型{row_type} vs After类型{col_type}")
        
        # 检查是否有跨类型相似度高于同类型
        if max_cross_type > np.min(diagonal):
            print(f"  ⚠️  警告: 存在跨类型相似度高于某些同类型相似度")
    
    # 显示完整的类型对应相似度
    print(f"\n完整相似度矩阵 (按类型):")
    header = "Before\\After  " + "  ".join([f"Type{t}" for t in mismatch_info['common_types']])
    print(header)
    for i, before_type in enumerate(mismatch_info['common_types']):
        row_str = f"Type{before_type}       "
        row_str += "  ".join([f"{similarity_matrix[i, j]:.4f}" for j in range(len(mismatch_info['common_types']))])
        print(row_str)

# 统计信息
print(f"\n\n{'='*80}")
print(f"处理摘要")
print(f"{'='*80}")
print(f"总场景数: {len(all_scene_dirs)}")
print(f"✓ 成功处理: {len(all_similarity_matrices)}")
print(f"⚠️  部分匹配: {len(partial_match_scenes)}")
print(f"❌ 完全失败: {len(failed_scenes)}")

# 输出失败场景详情
if failed_scenes:
    print(f"\n{'='*80}")
    print(f"❌ 完全失败的场景 ({len(failed_scenes)}个)")
    print(f"{'='*80}")
    for item in failed_scenes:
        print(f"\n场景: {item['scene']}")
        print(f"  错误: {item['error']}")
        if item['mismatch_info']:
            info = item['mismatch_info']
            if info.get('before_map'):
                print(f"  Before映射: {info['before_map']}")
            if info.get('after_map'):
                print(f"  After映射: {info['after_map']}")

# 输出部分匹配场景详情
if partial_match_scenes:
    print(f"\n{'='*80}")
    print(f"⚠️  部分匹配的场景 ({len(partial_match_scenes)}个)")
    print(f"{'='*80}")
    for item in partial_match_scenes:
        print(f"\n场景: {item['scene']}")
        info = item['mismatch_info']
        print(f"  Before映射: {info['before_map']}")
        print(f"  After映射: {info['after_map']}")
        print(f"  成功匹配的类型: {info['common_types']}")
        if info['only_before_types']:
            print(f"  只在Before中: {info['only_before_types']}")
        if info['only_after_types']:
            print(f"  只在After中: {info['only_after_types']}")
        if info['before_missing']:
            print(f"  Before缺失特征: {info['before_missing']}")
        if info['after_missing']:
            print(f"  After缺失特征: {info['after_missing']}")

# 统计相似度
if all_type_matched_similarities:
    print(f"\n{'='*80}")
    print(f"相似度统计 (基于 {len(all_similarity_matrices)} 个成功场景)")
    print(f"{'='*80}")
    
    print(f"\n相同类型相似度统计 (对角线元素):")
    print(f"  最小值: {np.min(all_type_matched_similarities):.6f}")
    print(f"  最大值: {np.max(all_type_matched_similarities):.6f}")
    print(f"  平均值: {np.mean(all_type_matched_similarities):.6f}")
    print(f"  中位数: {np.median(all_type_matched_similarities):.6f}")
    print(f"  标准差: {np.std(all_type_matched_similarities):.6f}")

if all_max_cross_type_similarities:
    print(f"\n跨类型最大相似度统计:")
    print(f"  最小值: {np.min(all_max_cross_type_similarities):.6f}")
    print(f"  最大值: {np.max(all_max_cross_type_similarities):.6f}")
    print(f"  平均值: {np.mean(all_max_cross_type_similarities):.6f}")
    print(f"  中位数: {np.median(all_max_cross_type_similarities):.6f}")
    print(f"  标准差: {np.std(all_max_cross_type_similarities):.6f}")
    
    # 分析跨类型混淆情况
    confusion_count = sum(1 for val in all_max_cross_type_similarities 
                         if val > np.mean(all_type_matched_similarities))
    print(f"\n跨类型相似度高于平均同类型相似度的场景数: {confusion_count}/{len(all_max_cross_type_similarities)}")

# 保存结果
output_file = "before_after_type_matched_similarity.npz"
np.savez(output_file,
         type_matched_similarities=np.array(all_type_matched_similarities),
         cross_type_similarities=np.array(all_max_cross_type_similarities))
print(f"\n结果已保存到: {output_file}")

# 保存详细报告
with open("before_after_type_matched_report.txt", "w", encoding='utf-8') as f:
    f.write("Before vs After 类型匹配相似度分析报告\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"处理摘要\n")
    f.write(f"-"*80 + "\n")
    f.write(f"总场景数: {len(all_scene_dirs)}\n")
    f.write(f"成功处理: {len(all_similarity_matrices)}\n")
    f.write(f"部分匹配: {len(partial_match_scenes)}\n")
    f.write(f"完全失败: {len(failed_scenes)}\n\n")
    
    # 失败场景
    if failed_scenes:
        f.write(f"\n完全失败的场景\n")
        f.write(f"="*80 + "\n")
        for item in failed_scenes:
            f.write(f"\n场景: {item['scene']}\n")
            f.write(f"错误: {item['error']}\n")
            if item['mismatch_info']:
                info = item['mismatch_info']
                if info.get('before_map'):
                    f.write(f"Before映射: {info['before_map']}\n")
                if info.get('after_map'):
                    f.write(f"After映射: {info['after_map']}\n")
    
    # 部分匹配场景
    if partial_match_scenes:
        f.write(f"\n部分匹配的场景\n")
        f.write(f"="*80 + "\n")
        for item in partial_match_scenes:
            f.write(f"\n场景: {item['scene']}\n")
            info = item['mismatch_info']
            f.write(f"Before映射: {info['before_map']}\n")
            f.write(f"After映射: {info['after_map']}\n")
            f.write(f"成功匹配: {info['common_types']}\n")
            if info['only_before_types']:
                f.write(f"只在Before中: {info['only_before_types']}\n")
            if info['only_after_types']:
                f.write(f"只在After中: {info['only_after_types']}\n")
    
    # 成功场景详情
    f.write(f"\n\n成功处理的场景详情\n")
    f.write(f"="*80 + "\n")
    for scene_name in sorted(all_similarity_matrices.keys()):
        data = all_similarity_matrices[scene_name]
        matrix = data['matrix']
        common_types = data['common_types']
        before_map = data['before_map']
        after_map = data['after_map']
        
        f.write(f"\n场景: {scene_name}\n")
        f.write("-"*80 + "\n")
        f.write(f"Before类型映射: {before_map}\n")
        f.write(f"After类型映射: {after_map}\n")
        f.write(f"匹配的类型: {common_types}\n\n")
        
        f.write("相似度矩阵:\n")
        f.write(str(matrix) + "\n\n")
        
        diagonal = np.diag(matrix)
        f.write("相同类型相似度:\n")
        for i, type_id in enumerate(common_types):
            f.write(f"  类型 {type_id}: {diagonal[i]:.6f}\n")
        
        f.write("\n")

print(f"详细报告已保存到: before_after_type_matched_report.txt")