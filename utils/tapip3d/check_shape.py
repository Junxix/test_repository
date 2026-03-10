import numpy as np
from pathlib import Path
from collections import defaultdict

def check_paired_first_dim_consistency(base_dir, camera_name='cam_104122063550'):
    """
    检查指定相机目录下配对文件的第一维形状是否一致：
    - robot_siglip 和 after_sam2_tapip3d_results_offline
    - human_siglip 和 before_sam2_tapip3d_results_offline
    """
    base_path = Path(base_dir)
    
    # 定义配对关系
    pairs = [
        {
            'name': 'robot配对',
            'siglip_dir': 'robot_siglip',
            'tracks_dir': 'after_sam2_tapip3d_results_offline',
            'siglip_pattern': 'target_{}.npy',
            'tracks_pattern': '3d_tracks_target_{}.npy'
        },
        {
            'name': 'human配对',
            'siglip_dir': 'human_siglip',
            'tracks_dir': 'before_sam2_tapip3d_results_offline',
            'siglip_pattern': 'target_{}.npy',
            'tracks_pattern': '3d_tracks_target_{}.npy'
        }
    ]
    
    # 遍历所有场景目录
    scene_dirs = [d for d in base_path.glob('task_*') if d.is_dir()]
    
    inconsistent_cases = []
    total_checks = 0
    consistent_checks = 0
    
    for scene_dir in sorted(scene_dirs):
        scene_name = scene_dir.name
        
        # 只检查指定的相机目录
        cam_dir = scene_dir / camera_name
        
        if not cam_dir.exists():
            print(f"场景 {scene_name}: ⚠️  {camera_name} 不存在，跳过")
            continue
        
        print(f"\n{'='*80}")
        print(f"场景: {scene_name}")
        print(f"相机: {camera_name}")
        
        # 检查每个配对
        for pair in pairs:
            print(f"\n  检查 {pair['name']}:")
            
            siglip_dir = cam_dir / pair['siglip_dir']
            tracks_dir = cam_dir / pair['tracks_dir']
            
            # 检查目录是否存在
            if not siglip_dir.exists():
                print(f"    ⚠️  {pair['siglip_dir']} 不存在")
                continue
            if not tracks_dir.exists():
                print(f"    ⚠️  {pair['tracks_dir']} 不存在")
                continue
            
            # 检查target_1到target_4
            for i in range(1, 5):
                siglip_file = siglip_dir / pair['siglip_pattern'].format(i)
                tracks_file = tracks_dir / pair['tracks_pattern'].format(i)
                
                # 检查文件是否都存在
                if not siglip_file.exists() and not tracks_file.exists():
                    continue
                
                if not siglip_file.exists():
                    print(f"    ⚠️  target_{i}: {pair['siglip_dir']} 文件不存在")
                    continue
                
                if not tracks_file.exists():
                    print(f"    ⚠️  target_{i}: {pair['tracks_dir']} 文件不存在")
                    continue
                
                try:
                    # 加载两个文件
                    siglip_arr = np.load(siglip_file)
                    tracks_arr = np.load(tracks_file)
                    
                    siglip_shape = siglip_arr.shape
                    tracks_shape = tracks_arr.shape
                    
                    total_checks += 1
                    
                    # 比较第一维
                    if siglip_shape[0] == tracks_shape[0]:
                        print(f"    ✓ target_{i}: 第一维一致 = {siglip_shape[0]}")
                        print(f"       {pair['siglip_dir']}: {siglip_shape}")
                        print(f"       {pair['tracks_dir']}: {tracks_shape}")
                        consistent_checks += 1
                    else:
                        print(f"    ✗ target_{i}: 第一维不一致!")
                        print(f"       {pair['siglip_dir']}: {siglip_shape} (第一维={siglip_shape[0]})")
                        print(f"       {pair['tracks_dir']}: {tracks_shape} (第一维={tracks_shape[0]})")
                        inconsistent_cases.append({
                            'scene': scene_name,
                            'camera': camera_name,
                            'pair': pair['name'],
                            'target': i,
                            'siglip_shape': siglip_shape,
                            'tracks_shape': tracks_shape
                        })
                
                except Exception as e:
                    print(f"    ❌ target_{i}: 加载失败 - {e}")
    
    # 总结
    print("\n" + "="*80)
    print("总结:")
    print(f"  总检查数: {total_checks}")
    print(f"  一致数: {consistent_checks}")
    print(f"  不一致数: {len(inconsistent_cases)}")
    
    if inconsistent_cases:
        print(f"\n发现 {len(inconsistent_cases)} 个不一致的情况:")
        for case in inconsistent_cases:
            print(f"\n  场景: {case['scene']}")
            print(f"  相机: {case['camera']}")
            print(f"  配对: {case['pair']}")
            print(f"  Target: {case['target']}")
            print(f"  SigLIP形状: {case['siglip_shape']} (第一维={case['siglip_shape'][0]})")
            print(f"  Tracks形状: {case['tracks_shape']} (第一维={case['tracks_shape'][0]})")
    else:
        print("\n✓ 所有配对文件的第一维形状都一致!")
    
    return inconsistent_cases

# 使用示例
base_dir = "/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train"
inconsistent = check_paired_first_dim_consistency(base_dir, camera_name='cam_104122063550')