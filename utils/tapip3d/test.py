import os
from pathlib import Path

# 基础路径
base_path = "/data/jingjing/data/context/realdata_sampled_20251110/train"

# 遍历scene_0021到scene_0099
for scene_num in range(21, 100):
    scene_name = f"scene_{scene_num:04d}"
    
    # 构建完整路径
    symlink_path = os.path.join(
        base_path,
        f"task_0103_user_0555_{scene_name}_cfg_0001",
        "cam_104122063550",
        "human_siglip",
        "human_siglip"
    )
    
    # 检查软链接是否存在
    if os.path.islink(symlink_path):
        try:
            os.unlink(symlink_path)
            print(f"✓ 已删除: {symlink_path}")
        except Exception as e:
            print(f"✗ 删除失败 {symlink_path}: {e}")
    elif os.path.exists(symlink_path):
        print(f"⚠ 警告: {symlink_path} 存在但不是软链接,跳过")
    else:
        print(f"- 不存在: {symlink_path}")

print("\n删除完成!")