import json
import os
from pathlib import Path

def find_robot_start_idx(color_dir, robot_start_time):
    """
    根据robot_start_time找到对应的图片索引
    
    Args:
        color_dir: color目录路径
        robot_start_time: robot开始时间戳(字符串)
    
    Returns:
        robot_start_idx: 对应的图片索引
    """
    # 获取所有png文件并提取时间戳
    png_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
    
    if not png_files:
        print(f"Warning: {color_dir} 中没有png文件")
        return None
    
    # 提取时间戳(去掉.png后缀)
    timestamps = [int(f.replace('.png', '')) for f in png_files]
    
    robot_time = int(robot_start_time)
    
    # 找到第一个大于等于robot_start_time的索引
    robot_start_idx = None
    for idx, ts in enumerate(timestamps):
        if ts >= robot_time:
            robot_start_idx = idx
            break
    
    # 如果没有找到大于等于的,说明robot_start_time在最后,使用最后一个索引
    if robot_start_idx is None:
        robot_start_idx = len(timestamps) - 1
    
    return robot_start_idx

def update_human_json(scene_dir):
    """
    更新单个scene的human.json文件
    
    Args:
        scene_dir: scene目录路径
    
    Returns:
        bool: 是否成功更新
    """
    human_json_path = os.path.join(scene_dir, 'human.json')
    color_dir = os.path.join(scene_dir, 'cam_104122063550', 'color')
    
    # 检查文件是否存在
    if not os.path.exists(human_json_path):
        print(f"  ⚠️  human.json 不存在")
        return False
    
    if not os.path.exists(color_dir):
        print(f"  ⚠️  color目录不存在")
        return False
    
    # 读取human.json
    try:
        with open(human_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ 读取human.json失败: {e}")
        return False
    
    # 获取robot_start_time (兼容两种可能的key名称)
    robot_start_time = data.get('robor_start_time') or data.get('robot_start_time')
    
    if not robot_start_time:
        print(f"  ⚠️  没有找到robot_start_time")
        return False
    
    # 计算robot_start_idx
    robot_start_idx = find_robot_start_idx(color_dir, robot_start_time)
    
    if robot_start_idx is None:
        return False
    
    # 更新数据
    data['robot_start_idx'] = robot_start_idx
    
    # 写回文件
    try:
        with open(human_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ 成功更新: robot_start_idx = {robot_start_idx}")
        return True
    except Exception as e:
        print(f"  ❌ 写入human.json失败: {e}")
        return False

def batch_update(root_dir):
    """
    批量更新所有scene的human.json
    
    Args:
        root_dir: 根目录路径
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ 目录不存在: {root_dir}")
        return
    
    # 遍历所有scene目录
    scene_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    print(f"找到 {len(scene_dirs)} 个场景目录\n")
    
    success_count = 0
    fail_count = 0
    
    for i, scene_dir in enumerate(scene_dirs, 1):
        print(f"[{i}/{len(scene_dirs)}] 处理: {scene_dir.name}")
        
        if update_human_json(str(scene_dir)):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n" + "="*60)
    print(f"处理完成!")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {fail_count}")
    print(f"  📊 总计: {len(scene_dirs)}")
    print("="*60)

if __name__ == "__main__":
    root_dir = "/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train"
    batch_update(root_dir)

