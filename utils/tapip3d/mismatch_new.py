import os
import json
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict

# 配置路径
source_base = Path("/data/jingjing/data/context/realdata_sampled_20251110/train")
output_base = Path("/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train")

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
        'robot_start_time': after_scene_info['robot_start_time'],
        'after_robot_start_idx': after_scene_info['camera']['robot_start_idx'],
        'before_robot_start_idx': before_scene_info['camera']['robot_start_idx']
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
            # 对TARGET_CAMERA进行特殊处理
            output_cam_dir.mkdir(parents=True, exist_ok=True)
            
            after_cam_info = after_scene_info['camera']
            before_cam_info = before_scene_info['camera']
            
            # 需要特殊处理的目录列表
            special_dirs = [
                "sam2_tapip3d_results_offline",
                "siglip_semantic_896_not_normalized"
            ]
            
            # 创建软链接 (除了特殊目录)
            after_cam_path = after_cam_info['cam_dir']
            for item in after_cam_path.iterdir():
                if item.name in special_dirs:
                    continue
                link_path = output_cam_dir / item.name
                if not link_path.exists():
                    os.symlink(item, link_path)
            
            # 处理sam2_tapip3d_results_offline
            before_robot_start_idx = before_cam_info['robot_start_idx']
            after_robot_start_idx = after_cam_info['robot_start_idx']
            
            before_sam2_dir = before_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
            after_sam2_dir = after_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
            
            if before_sam2_dir.exists() and after_sam2_dir.exists():
                create_split_sam2_results(
                    before_sam2_dir, after_sam2_dir,
                    before_robot_start_idx, after_robot_start_idx,
                    output_cam_dir / "before_sam2_tapip3d_results_offline",
                    output_cam_dir / "after_sam2_tapip3d_results_offline"
                )
            else:
                print(f"    警告: sam2_tapip3d_results_offline 目录不完整")
            
            # 处理siglip_semantic_896_not_normalized (直接复制)
            before_siglip_dir = before_cam_info['cam_dir'] / "siglip_semantic_896_not_normalized"
            after_siglip_dir = after_cam_info['cam_dir'] / "siglip_semantic_896_not_normalized"
            
            if before_siglip_dir.exists():
                copy_directory(
                    before_siglip_dir,
                    output_cam_dir / "before_siglip_semantic_896_not_normalized"
                )
            else:
                print(f"    警告: before siglip_semantic_896_not_normalized 不存在")
            
            if after_siglip_dir.exists():
                copy_directory(
                    after_siglip_dir,
                    output_cam_dir / "after_siglip_semantic_896_not_normalized"
                )
            else:
                print(f"    警告: after siglip_semantic_896_not_normalized 不存在")
        else:
            # 其他相机直接软链接
            if not output_cam_dir.exists():
                os.symlink(cam_dir, output_cam_dir)

def create_split_sam2_results(before_sam2_dir, after_sam2_dir, 
                               before_split_idx, after_split_idx,
                               output_before_dir, output_after_dir):
    """分割并保存sam2结果"""
    output_before_dir.mkdir(parents=True, exist_ok=True)
    output_after_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取target_type.json
    with open(before_sam2_dir / "target_type.json", 'r') as f:
        before_target_types = json.load(f)
    
    with open(after_sam2_dir / "target_type.json", 'r') as f:
        after_target_types = json.load(f)
    
    # 保存target_type.json
    with open(output_before_dir / "target_type.json", 'w') as f:
        json.dump(before_target_types, f, indent=2)
    
    with open(output_after_dir / "target_type.json", 'w') as f:
        json.dump(after_target_types, f, indent=2)
    
    # 处理before的每个target (取人类演示部分: [0:before_split_idx])
    for target_name in before_target_types.keys():
        target_num = target_name.replace('target', '')
        
        # 复制mask (不需要分割)
        mask_file = f"mask_target_{target_num}.png"
        mask_path = before_sam2_dir / mask_file
        if mask_path.exists():
            shutil.copy(mask_path, output_before_dir / mask_file)
        
        # 分割3d_tracks
        split_3d_tracks(
            before_sam2_dir / f"3d_tracks_target_{target_num}.npy",
            output_before_dir / f"3d_tracks_target_{target_num}.npy",
            0, before_split_idx
        )
        
        # 分割complete_result
        split_complete_result(
            before_sam2_dir / f"complete_result_target_{target_num}.npz",
            output_before_dir / f"complete_result_target_{target_num}.npz",
            0, before_split_idx
        )
        
        # 分割visibility
        split_visibility(
            before_sam2_dir / f"visibility_target_{target_num}.npy",
            output_before_dir / f"visibility_target_{target_num}.npy",
            0, before_split_idx
        )
    
    # 处理after的每个target (取机器人执行部分: [after_split_idx:])
    for target_name in after_target_types.keys():
        target_num = target_name.replace('target', '')
        
        # 复制mask
        mask_file = f"mask_target_{target_num}.png"
        mask_path = after_sam2_dir / mask_file
        if mask_path.exists():
            shutil.copy(mask_path, output_after_dir / mask_file)
        
        # 分割3d_tracks
        split_3d_tracks(
            after_sam2_dir / f"3d_tracks_target_{target_num}.npy",
            output_after_dir / f"3d_tracks_target_{target_num}.npy",
            after_split_idx, None
        )
        
        # 分割complete_result
        split_complete_result(
            after_sam2_dir / f"complete_result_target_{target_num}.npz",
            output_after_dir / f"complete_result_target_{target_num}.npz",
            after_split_idx, None
        )
        
        # 分割visibility
        split_visibility(
            after_sam2_dir / f"visibility_target_{target_num}.npy",
            output_after_dir / f"visibility_target_{target_num}.npy",
            after_split_idx, None
        )

def split_3d_tracks(input_file, output_file, start_idx, end_idx):
    """分割3d_tracks数据"""
    if not input_file.exists():
        print(f"    警告: {input_file} 不存在")
        return
    
    data = np.load(input_file)
    split_data = data[start_idx:end_idx]
    np.save(output_file, split_data)

def split_complete_result(input_file, output_file, start_idx, end_idx):
    """分割complete_result数据 - 只分割coords和visibs"""
    if not input_file.exists():
        print(f"    警告: {input_file} 不存在")
        return
    
    data = np.load(input_file)
    split_data = {}
    
    for key in data.keys():
        if key in ['coords', 'visibs']:
            # 只分割coords和visibs
            split_data[key] = data[key][start_idx:end_idx]
        else:
            # 其他字段保持不变
            split_data[key] = data[key]
    
    np.savez(output_file, **split_data)

def split_visibility(input_file, output_file, start_idx, end_idx):
    """分割visibility数据"""
    if not input_file.exists():
        print(f"    警告: {input_file} 不存在")
        return
    
    data = np.load(input_file)
    split_data = data[start_idx:end_idx]
    np.save(output_file, split_data)

def copy_directory(source_dir, output_dir):
    """复制整个目录"""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    shutil.copytree(source_dir, output_dir)

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