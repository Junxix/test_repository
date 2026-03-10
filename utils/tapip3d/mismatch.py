import os
import json
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict

# 配置路径
source_base = Path("/data/jingjing/data/context/realdata_sampled_20251110/train")
output_base = Path("/data/jingjing/data/context/realdata_sampled_mismatch/train")

# 指定需要处理的相机
TARGET_CAMERA = "cam_104122063550"

def load_scene_info(scene_path):
    """加载scene的基本信息"""
    # 读取human.json获取robor_start_time
    human_json_path = scene_path / "human.json"
    with open(human_json_path, 'r') as f:
        human_info = json.load(f)
    robot_start_time = human_info['robor_start_time']
    
    scene_info = {
        'path': scene_path,
        'robot_start_time': robot_start_time,
        'human_info': human_info
    }
    
    # 只读取TARGET_CAMERA的信息
    cam_dir = scene_path / TARGET_CAMERA
    if not cam_dir.exists():
        raise ValueError(f"Camera {TARGET_CAMERA} not found in {scene_path}")
    
    type_json_path = cam_dir / "type.json"
    target_type_json_path = cam_dir / "sam2_tapip3d_results_offline" / "target_type.json"
    
    with open(type_json_path, 'r') as f:
        cam_type = json.load(f)['type']
    
    with open(target_type_json_path, 'r') as f:
        target_types = json.load(f)
    
    # 获取时间戳列表
    color_dir = cam_dir / "color"
    timestamps = sorted([int(f.stem) for f in color_dir.glob("*.png")])
    
    # 找到robot_start_time对应的索引
    robot_start_idx = None
    for idx, ts in enumerate(timestamps):
        if str(ts) >= robot_start_time:
            robot_start_idx = idx
            break
    
    scene_info['camera'] = {
        'type': cam_type,
        'target_types': target_types,
        'timestamps': timestamps,
        'robot_start_idx': robot_start_idx,
        'cam_dir': cam_dir
    }
    
    return scene_info

def group_scenes_by_type(scenes_info):
    """按照type分组scenes"""
    type_groups = defaultdict(list)
    for scene_name, scene_info in scenes_info.items():
        scene_type = scene_info['camera']['type']
        type_groups[scene_type].append(scene_name)
    return type_groups

def create_mixed_scene(before_scene_info, after_scene_info, output_scene_path):
    """创建混合scene"""
    output_scene_path.mkdir(parents=True, exist_ok=True)
    
    # 创建混合记录json
    mix_record = {
        'before_scene': str(before_scene_info['path'].name),
        'after_scene': str(after_scene_info['path'].name),
        'robot_start_time': after_scene_info['robot_start_time']
    }
    
    with open(output_scene_path / "mix_record.json", 'w') as f:
        json.dump(mix_record, f, indent=2)
    
    # 复制human.json (使用after_scene的)
    shutil.copy(
        after_scene_info['path'] / "human.json",
        output_scene_path / "human.json"
    )
    
    # 复制metadata.json (使用after_scene的)
    metadata_path = after_scene_info['path'] / "metadata.json"
    if metadata_path.exists():
        shutil.copy(metadata_path, output_scene_path / "metadata.json")
    
    # 软链接其他文件和目录(如果存在)
    for item in after_scene_info['path'].iterdir():
        if item.name in ['human.json', 'metadata.json']:
            continue  # 已经复制过了
        if item.name.startswith('cam_'):
            continue  # 相机目录单独处理
        
        link_path = output_scene_path / item.name
        if not link_path.exists():
            os.symlink(item, link_path)
    
    # 处理所有相机目录
    all_cameras = [d for d in after_scene_info['path'].iterdir() 
                   if d.is_dir() and d.name.startswith('cam_')]
    
    for cam_dir in all_cameras:
        cam_name = cam_dir.name
        output_cam_dir = output_scene_path / cam_name
        
        if cam_name == TARGET_CAMERA:
            # 对TARGET_CAMERA进行混合处理
            output_cam_dir.mkdir(parents=True, exist_ok=True)
            
            after_cam_info = after_scene_info['camera']
            before_cam_info = before_scene_info['camera']
            
            # 创建软链接 (除了sam2_tapip3d_results_offline)
            after_cam_path = after_cam_info['cam_dir']
            for item in after_cam_path.iterdir():
                if item.name == "sam2_tapip3d_results_offline":
                    continue
                link_path = output_cam_dir / item.name
                if not link_path.exists():
                    os.symlink(item, link_path)
            
            # 创建混合的sam2_tapip3d_results_offline
            create_mixed_sam2_results(
                before_cam_info, after_cam_info, 
                output_cam_dir / "sam2_tapip3d_results_offline"
            )
        else:
            # 其他相机直接软链接
            if not output_cam_dir.exists():
                os.symlink(cam_dir, output_cam_dir)

def create_mixed_sam2_results(before_cam_info, after_cam_info, output_dir):
    """创建混合的sam2结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    before_sam2_dir = before_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
    after_sam2_dir = after_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
    
    # 复制target_type.json (使用after的)
    shutil.copy(
        after_sam2_dir / "target_type.json",
        output_dir / "target_type.json"
    )
    
    # 读取target_type映射
    with open(after_sam2_dir / "target_type.json", 'r') as f:
        after_target_types = json.load(f)
    
    with open(before_sam2_dir / "target_type.json", 'r') as f:
        before_target_types = json.load(f)
    
    # 创建type到target的反向映射
    before_type_to_target = {v: k for k, v in before_target_types.items()}
    
    robot_start_idx = after_cam_info['robot_start_idx']
    
    # 处理每个target
    for after_target_name, target_type in after_target_types.items():
        # 找到before_scene中相同type的target
        before_target_name = before_type_to_target.get(target_type)
        
        if before_target_name is None:
            print(f"    Warning: type {target_type} not found in before scene")
            continue
        
        # 提取target编号
        after_target_num = after_target_name.replace('target', '')
        before_target_num = before_target_name.replace('target', '')
        
        # 处理3d_tracks
        mix_3d_tracks(
            before_sam2_dir, after_sam2_dir,
            before_target_num, after_target_num,
            robot_start_idx, output_dir
        )
        
        # 处理complete_result
        mix_complete_result(
            before_sam2_dir, after_sam2_dir,
            before_target_num, after_target_num,
            robot_start_idx, output_dir
        )
        
        # 处理visibility
        mix_visibility(
            before_sam2_dir, after_sam2_dir,
            before_target_num, after_target_num,
            robot_start_idx, output_dir
        )
        
        # 复制mask (使用after的)
        mask_file = f"mask_target_{after_target_num}.png"
        shutil.copy(
            after_sam2_dir / mask_file,
            output_dir / mask_file
        )

def mix_3d_tracks(before_dir, after_dir, before_num, after_num, split_idx, output_dir):
    """混合3d_tracks数据"""
    before_file = before_dir / f"3d_tracks_target_{before_num}.npy"
    after_file = after_dir / f"3d_tracks_target_{after_num}.npy"
    
    before_data = np.load(before_file)
    after_data = np.load(after_file)
    
    # 拼接: before的前split_idx帧 + after的split_idx之后的帧
    mixed_data = np.concatenate([
        before_data[:split_idx],
        after_data[split_idx:]
    ], axis=0)
    
    output_file = output_dir / f"3d_tracks_target_{after_num}.npy"
    np.save(output_file, mixed_data)

def mix_complete_result(before_dir, after_dir, before_num, after_num, split_idx, output_dir):
    """混合complete_result数据"""
    before_file = before_dir / f"complete_result_target_{before_num}.npz"
    after_file = after_dir / f"complete_result_target_{after_num}.npz"
    
    before_data = np.load(before_file)
    after_data = np.load(after_file)
    
    # 混合coords (和3d_tracks一样)
    mixed_coords = np.concatenate([
        before_data['coords'][:split_idx],
        after_data['coords'][split_idx:]
    ], axis=0)
    
    # 混合其他字段
    mixed_data = {}
    for key in after_data.keys():
        if key == 'coords':
            mixed_data[key] = mixed_coords
        elif key in ['visibs', 'mask']:
            # 这些也需要按帧拼接
            mixed_data[key] = np.concatenate([
                before_data[key][:split_idx],
                after_data[key][split_idx:]
            ], axis=0)
        else:
            # video, depths, intrinsics, extrinsics等也需要拼接
            if len(before_data[key].shape) > 0 and before_data[key].shape[0] == before_data['coords'].shape[0]:
                mixed_data[key] = np.concatenate([
                    before_data[key][:split_idx],
                    after_data[key][split_idx:]
                ], axis=0)
            else:
                # 如果不是按帧的数据,使用after的
                mixed_data[key] = after_data[key]
    
    output_file = output_dir / f"complete_result_target_{after_num}.npz"
    np.savez(output_file, **mixed_data)

def mix_visibility(before_dir, after_dir, before_num, after_num, split_idx, output_dir):
    """混合visibility数据"""
    before_file = before_dir / f"visibility_target_{before_num}.npy"
    after_file = after_dir / f"visibility_target_{after_num}.npy"
    
    before_data = np.load(before_file)
    after_data = np.load(after_file)
    
    mixed_data = np.concatenate([
        before_data[:split_idx],
        after_data[split_idx:]
    ], axis=0)
    
    output_file = output_dir / f"visibility_target_{after_num}.npy"
    np.save(output_file, mixed_data)

def main():
    # 收集所有scenes
    all_scenes = sorted([d for d in source_base.iterdir() if d.is_dir()])
    
    print(f"找到 {len(all_scenes)} 个scenes")
    
    # 加载所有scene信息
    scenes_info = {}
    for scene_path in all_scenes:
        scene_name = scene_path.name
        try:
            scene_info = load_scene_info(scene_path)
            scenes_info[scene_name] = scene_info
            print(f"加载scene: {scene_name}")
        except Exception as e:
            print(f"跳过scene {scene_name}: {e}")
    
    # 按type分组
    type_groups = group_scenes_by_type(scenes_info)
    print(f"\nScene类型分组:")
    for scene_type, scene_list in type_groups.items():
        print(f"  Type {scene_type}: {len(scene_list)} scenes")
    
    # 对每个type组内的scenes进行混合
    mixed_count = 0
    for scene_type, scene_list in type_groups.items():
        if len(scene_list) < 2:
            print(f"\nType {scene_type} 只有1个scene,跳过")
            continue
        
        print(f"\n处理Type {scene_type}的scenes...")
        
        # 每个scene作为after_scene,与组内其他scenes作为before_scene进行混合
        for i, after_scene_name in enumerate(scene_list):
            after_scene_info = scenes_info[after_scene_name]
            
            for j, before_scene_name in enumerate(scene_list):
                if i == j:
                    continue  # 跳过自己
                
                before_scene_info = scenes_info[before_scene_name]
                
                # 创建新的scene名称
                new_scene_name = f"{before_scene_name}_BEFORE_{after_scene_name}_AFTER"
                output_scene_path = output_base / new_scene_name
                
                print(f"  创建混合scene: {new_scene_name}")
                
                try:
                    create_mixed_scene(before_scene_info, after_scene_info, output_scene_path)
                    mixed_count += 1
                except Exception as e:
                    print(f"    错误: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n完成! 共创建 {mixed_count} 个混合scenes")

if __name__ == "__main__":
    main()