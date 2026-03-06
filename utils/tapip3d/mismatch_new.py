import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

# 配置路径
source_base = Path("/data/jingjing/data/context/realdata_sampled_20260109/train")
output_base = Path("/data/jingjing/data/context/realdata_sampled_20260109_mismatch/train")

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
    
    with open(type_json_path, 'r') as f:
        cam_type = json.load(f)['type']
    
    scene_info['camera'] = {
        'type': cam_type,
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
            # 对TARGET_CAMERA进行特殊处理
            output_cam_dir.mkdir(parents=True, exist_ok=True)
            
            after_cam_info = after_scene_info['camera']
            before_cam_info = before_scene_info['camera']
            
            # 需要特殊处理的目录列表
            special_dirs = ["sam2_tapip3d_results_offline"]
            
            # 分别处理 before 和 after 的相机目录
            after_cam_path = after_cam_info['cam_dir']
            before_cam_path = before_cam_info['cam_dir']
            
            # 从 after_cam_path 链接 robot 相关和共享的内容
            for item in after_cam_path.iterdir():
                if item.name in special_dirs:
                    continue
                # 跳过 human 相关的内容
                if 'human' in item.name.lower():
                    continue
                    
                link_path = output_cam_dir / item.name
                if not link_path.exists():
                    os.symlink(item, link_path)
            
            # 从 before_cam_path 链接 human 相关的内容
            for item in before_cam_path.iterdir():
                if item.name in special_dirs:
                    continue
                # 只链接 human 相关的内容
                if 'human' in item.name.lower():
                    link_path = output_cam_dir / item.name
                    if not link_path.exists():
                        os.symlink(item, link_path)
            
            # 处理sam2_tapip3d_results_offline
            before_sam2_dir = before_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
            after_sam2_dir = after_cam_info['cam_dir'] / "sam2_tapip3d_results_offline"
            
            if before_sam2_dir.exists() and after_sam2_dir.exists():
                copy_sam2_results(
                    before_sam2_dir, after_sam2_dir,
                    output_cam_dir / "before_sam2_tapip3d_results_offline",
                    output_cam_dir / "after_sam2_tapip3d_results_offline"
                )
            else:
                print(f"    警告: sam2_tapip3d_results_offline 目录不完整")
        else:
            # 其他相机直接软链接
            if not output_cam_dir.exists():
                os.symlink(cam_dir, output_cam_dir)

def copy_sam2_results(before_sam2_dir, after_sam2_dir, output_before_dir, output_after_dir):
    """复制sam2结果文件"""
    output_before_dir.mkdir(parents=True, exist_ok=True)
    output_after_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制before相关的所有文件
    for file in before_sam2_dir.glob("*_before_*"):
        shutil.copy(file, output_before_dir / file.name)
    
    # 复制after相关的所有文件
    for file in after_sam2_dir.glob("*_after_*"):
        shutil.copy(file, output_after_dir / file.name)

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
        print(f"\n处理Type {scene_type}的scenes...")
        
        # 每个scene作为after_scene,与组内所有scenes(包括自己)作为before_scene进行混合
        for i, after_scene_name in enumerate(scene_list):
            after_scene_info = scenes_info[after_scene_name]
            
            for j, before_scene_name in enumerate(scene_list):
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