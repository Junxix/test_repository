import os
import json

def delete_files_in_time_ranges(directory_path, start_time1, finish_time1, start_time2, finish_time2):
    """
    删除指定目录中在时间范围内的PNG文件
    """
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return
    
    files = os.listdir(directory_path)
    deleted_count = 0
    
    for file in files:
        if file.endswith(".png"):
            try:
                # 从文件名中提取时间戳
                file_timestamp = int(file.split('.')[0])
                
                # 检查是否在指定时间范围内
                if (start_time1 <= file_timestamp <= finish_time1) or (start_time2 <= file_timestamp <= finish_time2):
                    file_path = os.path.join(directory_path, file)
                    os.remove(file_path)
                    # print(f"已删除: {file}")
                    deleted_count += 1
            except ValueError:
                print(f"无法解析时间戳: {file}")
    
    print(f"从 {directory_path} 中删除了 {deleted_count} 个文件")

def process_single_scene(base_dir, scene_name):
    """
    处理单个场景目录
    """
    scene_path = os.path.join(base_dir, scene_name)
    json_file_path = os.path.join(scene_path, "human.json")
    
    print(f"\n处理场景: {scene_name}")
    
    # 检查human.json文件是否存在
    if not os.path.exists(json_file_path):
        print(f"跳过 {scene_name}: human.json 文件不存在")
        return
    
    try:
        # 读取JSON数据
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # 提取时间值
        start_time1 = int(data.get("start_time1"))
        finish_time1 = int(data.get("finish_time1"))
        start_time2 = int(data.get("start_time2"))
        finish_time2 = int(data.get("finish_time2"))
        
        print(f"时间范围1: {start_time1} - {finish_time1}")
        print(f"时间范围2: {start_time2} - {finish_time2}")
        
        # 定义相机目录列表
        camera_dirs = [
            "cam_750612070851",
            "cam_043322070878", 
            "cam_104122063550"
        ]
        
        # 处理每个相机目录的color和depth文件夹
        for camera_dir in camera_dirs:
            for sub_dir in ["color", "depth"]:
                directory_path = os.path.join(scene_path, camera_dir, sub_dir)
                print(f"  处理目录: {camera_dir}/{sub_dir}")
                delete_files_in_time_ranges(
                    directory_path, 
                    start_time1, finish_time1, 
                    start_time2, finish_time2
                )
    
    except Exception as e:
        print(f"处理场景 {scene_name} 时出错: {e}")

def main():
    """
    主函数：批量处理多个场景
    """
    # 基础目录
    base_dir = "/home/ubuntu/data/realdata_20260202_val"
    
    # 生成场景列表（scene_0002 到 scene_0035）
    scenes = [f"task_0108_user_0555_scene_{i:04d}_cfg_0001" for i in range(4,5)]
    
    print(f"准备处理 {len(scenes)} 个场景...")
    
    # 处理每个场景
    for scene in scenes:
        process_single_scene(base_dir, scene)
    
    print("\n所有场景处理完成！")

if __name__ == "__main__":
    main()