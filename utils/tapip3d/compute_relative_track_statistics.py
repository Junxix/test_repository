"""
计算数据集中相对运动(tracks)的统计范围
用于确定REL_TRACK_MIN和REL_TRACK_MAX
支持添加安全margin和计算覆盖率
主要使用0.5%-99.5%百分位数
"""

import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_relative_track_stats(data_path, cam_ids=['104122063550'], num_targets=2, split='train'):
    """
    遍历数据集,计算相对运动的统计信息
    """
    data_split_path = os.path.join(data_path, split)
    all_demos = sorted(os.listdir(data_split_path))
    
    all_relative_movements = []
    
    print(f"Processing {len(all_demos)} demos...")
    
    for demo_name in tqdm(all_demos):
        demo_path = os.path.join(data_split_path, demo_name)
        
        for cam_id in cam_ids:
            # Process human tracks (before)
            cam_path = os.path.join(demo_path, f"cam_{cam_id}")
            human_tracks_dir = os.path.join(cam_path, "before_sam2_tapip3d_results_offline")
            
            if not os.path.exists(human_tracks_dir):
                continue
            
            # Load human tracks
            for target_idx in range(1, num_targets + 1):
                human_target_path = os.path.join(human_tracks_dir, f"3d_tracks_target_{target_idx}.npy")
                
                if not os.path.exists(human_target_path):
                    continue
                
                try:
                    tracks = np.load(human_target_path)
                    if tracks.ndim == 4 and num_targets == 1:
                        tracks = tracks[0]
                    
                    # 计算相对运动
                    first_frame = tracks[0:1]  # (1, num_points, 3)
                    relative_tracks = tracks - first_frame  # (T, num_points, 3)
                    
                    # 收集所有相对运动值
                    all_relative_movements.append(relative_tracks.reshape(-1, 3))
                    
                except Exception as e:
                    print(f"Error processing {human_target_path}: {e}")
                    continue
            
            # Process robot tracks (after)
            robot_tracks_dir = os.path.join(cam_path, "after_sam2_tapip3d_results_offline")
            
            if not os.path.exists(robot_tracks_dir):
                continue
            
            # Load target_type mapping
            try:
                robot_target_type_path = os.path.join(robot_tracks_dir, "target_type.json")
                with open(robot_target_type_path, 'r') as f:
                    robot_target_types = json.load(f)
            except:
                continue
            
            # Load robot tracks
            for target_key in robot_target_types.keys():
                target_idx = int(target_key.replace('target', ''))
                robot_target_path = os.path.join(robot_tracks_dir, f"3d_tracks_target_{target_idx}.npy")
                
                if not os.path.exists(robot_target_path):
                    continue
                
                try:
                    tracks = np.load(robot_target_path)
                    if tracks.ndim == 4 and num_targets == 1:
                        tracks = tracks[0]
                    
                    # 计算相对运动
                    first_frame = tracks[0:1]
                    relative_tracks = tracks - first_frame
                    
                    all_relative_movements.append(relative_tracks.reshape(-1, 3))
                    
                except Exception as e:
                    print(f"Error processing {robot_target_path}: {e}")
                    continue
    
    if len(all_relative_movements) == 0:
        raise ValueError("No valid tracks found!")
    
    # 合并所有相对运动
    all_relative_movements = np.concatenate(all_relative_movements, axis=0)  # (N, 3)
    
    print(f"\nTotal relative movement points: {all_relative_movements.shape[0]}")
    
    # 计算统计信息
    stats = {}
    
    # 基本统计
    stats['min'] = all_relative_movements.min(axis=0)
    stats['max'] = all_relative_movements.max(axis=0)
    stats['mean'] = all_relative_movements.mean(axis=0)
    stats['std'] = all_relative_movements.std(axis=0)
    stats['median'] = np.median(all_relative_movements, axis=0)
    
    # 百分位数 - 重点关注0.5和99.5
    for percentile in [0.1, 0.5, 1, 5, 95, 99, 99.5, 99.9]:
        stats[f'p{percentile}'] = np.percentile(all_relative_movements, percentile, axis=0)
    
    return stats, all_relative_movements


def calculate_coverage(all_movements, range_min, range_max):
    """计算范围覆盖率"""
    in_range = (all_movements >= range_min) & (all_movements <= range_max)
    in_range_all = in_range.all(axis=1)  # 所有维度都在范围内
    coverage = in_range_all.sum() / len(all_movements) * 100
    return coverage


def calculate_clipped_points(all_movements, range_min, range_max):
    """计算会被clip的点数"""
    out_of_range = (all_movements < range_min) | (all_movements > range_max)
    out_of_range_any = out_of_range.any(axis=1)  # 任意维度超出范围
    clipped_count = out_of_range_any.sum()
    clipped_percentage = clipped_count / len(all_movements) * 100
    return clipped_count, clipped_percentage


def print_statistics(stats, all_movements, margin=0.2):
    """
    打印统计信息
    margin: 安全边界,例如0.2表示在百分位数基础上扩展20%
    """
    print("\n" + "="*80)
    print("相对运动统计 (Relative Track Statistics)")
    print("="*80)
    
    axes = ['X', 'Y', 'Z']
    
    print(f"\n{'Statistic':<15} {'X (m)':<15} {'Y (m)':<15} {'Z (m)':<15}")
    print("-" * 60)
    
    for key in ['min', 'p0.1', 'p0.5', 'p1', 'p5', 'mean', 'median', 'p95', 'p99', 'p99.5', 'p99.9', 'max', 'std']:
        values = stats[key]
        print(f"{key:<15} {values[0]:>14.4f} {values[1]:>14.4f} {values[2]:>14.4f}")
    
    print("\n" + "="*80)
    print("推荐的 REL_TRACK_MIN 和 REL_TRACK_MAX (带覆盖率分析):")
    print("="*80)
    
    # 方法1: 使用0.5%-99.5%百分位数 (推荐)
    rel_min_p995 = stats['p0.5']
    rel_max_p995 = stats['p99.5']
    coverage_p995 = calculate_coverage(all_movements, rel_min_p995, rel_max_p995)
    clipped_p995, clipped_pct_p995 = calculate_clipped_points(all_movements, rel_min_p995, rel_max_p995)
    
    print(f"\n方法1: 使用0.5%-99.5%百分位数 (无margin) ⭐推荐")
    print(f"REL_TRACK_MIN = np.array([{rel_min_p995[0]:.4f}, {rel_min_p995[1]:.4f}, {rel_min_p995[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_p995[0]:.4f}, {rel_max_p995[1]:.4f}, {rel_max_p995[2]:.4f}])")
    print(f"覆盖率: {coverage_p995:.2f}%")
    print(f"会被clip的点数: {clipped_p995} ({clipped_pct_p995:.2f}%)")
    
    # 方法2: 使用0.5%-99.5%百分位数 + margin
    rel_min_with_margin = stats['p0.5'] * (1 + margin * np.sign(stats['p0.5']))
    rel_max_with_margin = stats['p99.5'] * (1 + margin * np.sign(stats['p99.5']))
    coverage_margin = calculate_coverage(all_movements, rel_min_with_margin, rel_max_with_margin)
    clipped_margin, clipped_pct_margin = calculate_clipped_points(all_movements, rel_min_with_margin, rel_max_with_margin)
    
    print(f"\n方法2: 使用0.5%-99.5%百分位数 + {margin*100}% margin")
    print(f"REL_TRACK_MIN = np.array([{rel_min_with_margin[0]:.4f}, {rel_min_with_margin[1]:.4f}, {rel_min_with_margin[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_with_margin[0]:.4f}, {rel_max_with_margin[1]:.4f}, {rel_max_with_margin[2]:.4f}])")
    print(f"覆盖率: {coverage_margin:.2f}%")
    print(f"会被clip的点数: {clipped_margin} ({clipped_pct_margin:.2f}%)")
    
    # 方法3: 使用1%-99%百分位数
    rel_min_p99 = stats['p1']
    rel_max_p99 = stats['p99']
    coverage_p99 = calculate_coverage(all_movements, rel_min_p99, rel_max_p99)
    clipped_p99, clipped_pct_p99 = calculate_clipped_points(all_movements, rel_min_p99, rel_max_p99)
    
    print(f"\n方法3: 使用1%-99%百分位数 (更保守)")
    print(f"REL_TRACK_MIN = np.array([{rel_min_p99[0]:.4f}, {rel_min_p99[1]:.4f}, {rel_min_p99[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_p99[0]:.4f}, {rel_max_p99[1]:.4f}, {rel_max_p99[2]:.4f}])")
    print(f"覆盖率: {coverage_p99:.2f}%")
    print(f"会被clip的点数: {clipped_p99} ({clipped_pct_p99:.2f}%)")
    
    # 方法4: 使用5%-95%百分位数
    rel_min_p95 = stats['p5']
    rel_max_p95 = stats['p95']
    coverage_p95 = calculate_coverage(all_movements, rel_min_p95, rel_max_p95)
    clipped_p95, clipped_pct_p95 = calculate_clipped_points(all_movements, rel_min_p95, rel_max_p95)
    
    print(f"\n方法4: 使用5%-95%百分位数 (非常保守)")
    print(f"REL_TRACK_MIN = np.array([{rel_min_p95[0]:.4f}, {rel_min_p95[1]:.4f}, {rel_min_p95[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_p95[0]:.4f}, {rel_max_p95[1]:.4f}, {rel_max_p95[2]:.4f}])")
    print(f"覆盖率: {coverage_p95:.2f}%")
    print(f"会被clip的点数: {clipped_p95} ({clipped_pct_p95:.2f}%)")
    
    # 方法5: 使用绝对最大值
    rel_min_abs = stats['min']
    rel_max_abs = stats['max']
    coverage_abs = calculate_coverage(all_movements, rel_min_abs, rel_max_abs)
    
    print(f"\n方法5: 使用绝对最小/最大值 (覆盖100%,可能有outliers)")
    print(f"REL_TRACK_MIN = np.array([{rel_min_abs[0]:.4f}, {rel_min_abs[1]:.4f}, {rel_min_abs[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_abs[0]:.4f}, {rel_max_abs[1]:.4f}, {rel_max_abs[2]:.4f}])")
    print(f"覆盖率: {coverage_abs:.2f}% (100%)")
    print(f"会被clip的点数: 0 (0.00%)")
    
    # 输出推荐
    print("\n" + "="*80)
    print("⭐ 推荐使用:")
    print("="*80)
    print("方法1: 使用0.5%-99.5%百分位数 (无margin)")
    print(f"REL_TRACK_MIN = np.array([{rel_min_p995[0]:.4f}, {rel_min_p995[1]:.4f}, {rel_min_p995[2]:.4f}])")
    print(f"REL_TRACK_MAX = np.array([{rel_max_p995[0]:.4f}, {rel_max_p995[1]:.4f}, {rel_max_p995[2]:.4f}])")
    print(f"\n覆盖率: {coverage_p995:.2f}% (覆盖99%的数据)")
    print(f"会被clip的点数: {clipped_p995} ({clipped_pct_p995:.2f}%)")
    print(f"\n在代码中使用 np.clip(normalized, -1.0, 1.0) 确保所有值在 [-1, 1] 范围内")
    print(f"预计有 {clipped_pct_p995:.2f}% 的数据点会被clip,这些通常是噪声或异常值")


def plot_distributions(all_movements, stats, save_path='relative_track_stats.png', margin=0.2):
    """绘制分布图"""
    fig = plt.figure(figsize=(20, 15))
    
    axes_names = ['X', 'Y', 'Z']
    
    # 使用0.5%-99.5%作为推荐范围
    rel_min_rec = stats['p0.5']
    rel_max_rec = stats['p99.5']
    
    # 1. 直方图 (每个轴)
    for i, axis_name in enumerate(axes_names):
        ax = plt.subplot(4, 3, i + 1)
        data = all_movements[:, i]
        ax.hist(data, bins=100, alpha=0.7, edgecolor='black', color='skyblue')
        
        # 标记关键统计值
        ax.axvline(stats['mean'][i], color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(stats['p0.5'][i], color='orange', linestyle='--', linewidth=2, label='0.5%')
        ax.axvline(stats['p99.5'][i], color='orange', linestyle='--', linewidth=2, label='99.5%')
        ax.axvline(rel_min_rec[i], color='green', linestyle='-', linewidth=2.5, label='Rec Min', alpha=0.8)
        ax.axvline(rel_max_rec[i], color='green', linestyle='-', linewidth=2.5, label='Rec Max', alpha=0.8)
        
        ax.set_xlabel(f'{axis_name} (m)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{axis_name}-axis Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 2. Box plot (所有轴)
    ax = plt.subplot(4, 3, 4)
    bp = ax.boxplot([all_movements[:, i] for i in range(3)], 
                     labels=axes_names,
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Relative Movement (m)', fontsize=12)
    ax.set_title('Box Plot (All Axes)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    for i, axis_name in enumerate(axes_names):
        ax = plt.subplot(4, 3, 5 + i)
        data = np.sort(all_movements[:, i])
        cumulative = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cumulative, linewidth=2, color='blue')
        ax.axvline(stats['p0.5'][i], color='orange', linestyle='--', alpha=0.7, linewidth=2, label='0.5%')
        ax.axvline(stats['p99.5'][i], color='orange', linestyle='--', alpha=0.7, linewidth=2, label='99.5%')
        ax.axvline(rel_min_rec[i], color='green', linestyle='-', alpha=0.8, linewidth=2.5, label='Rec Min')
        ax.axvline(rel_max_rec[i], color='green', linestyle='-', alpha=0.8, linewidth=2.5, label='Rec Max')
        
        # 标记百分位
        ax.axhline(0.005, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(0.995, color='orange', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f'{axis_name} (m)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title(f'{axis_name}-axis CDF', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 4. 3D scatter (采样显示)
    ax = plt.subplot(4, 3, 8, projection='3d')
    sample_size = min(10000, len(all_movements))
    sample_indices = np.random.choice(len(all_movements), sample_size, replace=False)
    sample_data = all_movements[sample_indices]
    
    # 区分在范围内和范围外的点
    in_range = ((sample_data >= rel_min_rec) & (sample_data <= rel_max_rec)).all(axis=1)
    
    ax.scatter(sample_data[in_range, 0], sample_data[in_range, 1], sample_data[in_range, 2], 
               alpha=0.3, s=1, c='blue', label=f'In range ({in_range.sum()/len(in_range)*100:.1f}%)')
    ax.scatter(sample_data[~in_range, 0], sample_data[~in_range, 1], sample_data[~in_range, 2], 
               alpha=0.6, s=3, c='red', label=f'Out of range ({(~in_range).sum()/len(in_range)*100:.1f}%)')
    
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(f'3D Scatter (sampled {sample_size} points)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    
    # 5. 2D scatter plots
    scatter_configs = [(0, 1, 9), (0, 2, 10), (1, 2, 11)]
    for i, j, subplot_idx in scatter_configs:
        ax = plt.subplot(4, 3, subplot_idx)
        sample_data_2d = all_movements[sample_indices]
        
        # 区分在范围内和范围外的点
        in_range_2d = ((sample_data_2d[:, i] >= rel_min_rec[i]) & 
                       (sample_data_2d[:, i] <= rel_max_rec[i]) &
                       (sample_data_2d[:, j] >= rel_min_rec[j]) & 
                       (sample_data_2d[:, j] <= rel_max_rec[j]))
        
        ax.scatter(sample_data_2d[in_range_2d, i], sample_data_2d[in_range_2d, j], 
                  alpha=0.3, s=1, c='blue', label='In range')
        ax.scatter(sample_data_2d[~in_range_2d, i], sample_data_2d[~in_range_2d, j], 
                  alpha=0.6, s=3, c='red', label='Out of range')
        
        # 画出推荐范围的矩形
        from matplotlib.patches import Rectangle
        rect = Rectangle((rel_min_rec[i], rel_min_rec[j]), 
                        rel_max_rec[i] - rel_min_rec[i],
                        rel_max_rec[j] - rel_min_rec[j],
                        linewidth=2.5, edgecolor='green', facecolor='none',
                        linestyle='-', label='Recommended range (0.5%-99.5%)')
        ax.add_patch(rect)
        
        ax.set_xlabel(f'{axes_names[i]} (m)', fontsize=10)
        ax.set_ylabel(f'{axes_names[j]} (m)', fontsize=10)
        ax.set_title(f'{axes_names[i]}-{axes_names[j]} Scatter', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 6. 统计摘要文本
    ax = plt.subplot(4, 3, 12)
    ax.axis('off')
    
    coverage = calculate_coverage(all_movements, rel_min_rec, rel_max_rec)
    clipped_count, clipped_pct = calculate_clipped_points(all_movements, rel_min_rec, rel_max_rec)
    
    summary_text = f"""
    ⭐ 推荐配置 (0.5%-99.5%百分位数):
    
    REL_TRACK_MIN:
      X: {rel_min_rec[0]:.4f} m
      Y: {rel_min_rec[1]:.4f} m
      Z: {rel_min_rec[2]:.4f} m
    
    REL_TRACK_MAX:
      X: {rel_max_rec[0]:.4f} m
      Y: {rel_max_rec[1]:.4f} m
      Z: {rel_max_rec[2]:.4f} m
    
    覆盖率: {coverage:.2f}%
    会被clip的点: {clipped_count:,} ({clipped_pct:.2f}%)
    
    总数据点: {len(all_movements):,}
    
    说明:
    - 使用0.5%-99.5%覆盖99%的数据
    - 只有1%的极端值会被clip
    - 比1%-99%多覆盖1%的数据
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Relative Track Statistics (0.5%-99.5% Percentile)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n分布图已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='计算相对运动(tracks)的统计范围(使用0.5%-99.5%百分位数)')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='数据集路径')
    parser.add_argument('--cam_ids', type=str, nargs='+', 
                        default=['104122063550'],
                        help='相机ID列表')
    parser.add_argument('--num_targets', type=int, default=2,
                        help='目标数量')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集split (train/val/all)')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='安全边界比例(可选),例如0.2表示在百分位数基础上扩展20%')
    parser.add_argument('--save_plot', type=str, default='relative_track_stats.png',
                        help='保存分布图的路径')
    
    args = parser.parse_args()
    
    # 计算统计信息
    stats, all_movements = compute_relative_track_stats(
        args.data_path,
        cam_ids=args.cam_ids,
        num_targets=args.num_targets,
        split=args.split
    )
    
    # 打印统计信息
    print_statistics(stats, all_movements, margin=args.margin)
    
    # 绘制分布图
    plot_distributions(all_movements, stats, args.save_plot, margin=args.margin)
    
    print("\n" + "="*80)
    print("✅ 完成!")
    print("="*80)
    print("请按照以下步骤操作:")
    print("1. 查看上述统计结果和生成的分布图")
    print("2. 将推荐的 REL_TRACK_MIN 和 REL_TRACK_MAX 复制到 utils/constants.py")
    print("3. 确保在 dataset/realworld.py 的加载函数中使用了:")
    print("   - 先计算相对差值: relative = tracks - tracks[0]")
    print("   - 再normalize: normalized = (relative - REL_MIN) / (REL_MAX - REL_MIN) * 2 - 1")
    print("   - 最后clip: normalized = np.clip(normalized, -1.0, 1.0)")
    print("4. 重新训练模型")
    print("\n推荐使用 0.5%-99.5% 百分位数,覆盖99%的数据!")


if __name__ == '__main__':
    main()