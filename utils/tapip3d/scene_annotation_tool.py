import cv2
import os
import json
import glob
import numpy as np
from pathlib import Path

class SceneAnnotationTool:
    def __init__(self, base_path, task_id, user_id, cam_id="cam_104122063550"):
        self.base_path = base_path
        self.task_id = task_id
        self.user_id = user_id
        self.cam_id = cam_id
        self.type_mapping = {
            '1': 'box',
            '2': 'block',
            '3': 'cup',
            '4': 'doll',
            '5': 'yellow'
        }
        
    def get_scene_path(self, scene_num):
        """获取场景路径"""
        # 路径格式: task_0102_user_0555_scene_0001_cfg_0001/cam_104122063550
        pattern = os.path.join(
            self.base_path, 
            f"{self.task_id}_{self.user_id}_scene_{scene_num:04d}_cfg_*",
            self.cam_id
        )
        matching_paths = glob.glob(pattern)
        
        if matching_paths:
            return matching_paths[0]
        return None
    
    def get_images(self, scene_path):
        """获取场景下的所有图片"""
        color_path = os.path.join(scene_path, "color")
        if not os.path.exists(color_path):
            return []
        
        images = sorted(glob.glob(os.path.join(color_path, "*.png")))
        return images
    
    def create_info_panel(self, width=800, height=200):
        """创建信息面板"""
        panel = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # 标题
        cv2.putText(panel, "Scene Type Annotation", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 类型选项
        y_start = 90
        types = [
            "1: box",
            "2: block", 
            "3: cup",
            "4: doll",
            "5: yellow"
        ]
        
        for i, type_text in enumerate(types):
            x = 50 + (i % 3) * 250
            y = y_start + (i // 3) * 40
            cv2.putText(panel, type_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
        
        # 操作说明
        cv2.putText(panel, "Press 1-5: Select type  |  N: Next scene  |  Q: Quit", 
                   (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        return panel
    
    def save_type_json(self, scene_path, type_value):
        """保存type.json文件"""
        json_path = os.path.join(scene_path, "type.json")
        data = {"type": int(type_value)}
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"已保存: {json_path}")
    
    def play_scene(self, scene_num):
        """播放场景视频并标注"""
        scene_path = self.get_scene_path(scene_num)
        
        if not scene_path:
            print(f"未找到场景 scene_{scene_num:04d}")
            return None
        
        images = self.get_images(scene_path)
        
        if not images:
            print(f"场景 scene_{scene_num:04d} 没有图片")
            return None
        
        print(f"\n正在播放场景: scene_{scene_num:04d}")
        print(f"路径: {scene_path}")
        print(f"图片数量: {len(images)}")
        
        # 检查是否已有标注
        json_path = os.path.join(scene_path, "type.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_type = json.load(f).get('type', None)
                if existing_type:
                    type_name = self.type_mapping.get(str(existing_type), 'unknown')
                    print(f"⚠️  已存在标注: type={existing_type} ({type_name})")
        
        # 创建窗口
        cv2.namedWindow('Scene Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Info Panel', cv2.WINDOW_NORMAL)
        
        info_panel = self.create_info_panel()
        cv2.imshow('Info Panel', info_panel)
        
        current_type = None
        frame_idx = 0
        fps = 600  # 播放速度
        
        while True:
            # 循环播放图片
            img_path = images[frame_idx % len(images)]
            img = cv2.imread(img_path)
            
            if img is not None:
                # 添加场景信息
                h, w = img.shape[:2]
                overlay = img.copy()
                
                # 半透明背景
                cv2.rectangle(overlay, (10, 10), (w-10, 80), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                
                # 场景信息
                cv2.putText(img, f"Scene: scene_{scene_num:04d}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img, f"Frame: {frame_idx+1}/{len(images)}", (20, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 如果已选择类型，显示
                if current_type:
                    type_name = self.type_mapping[current_type]
                    cv2.putText(img, f"Selected: {current_type} ({type_name})", 
                               (w-300, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Scene Video', img)
            
            # 控制播放速度
            key = cv2.waitKey(1000 // fps) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return 'quit'
            elif key == ord('n'):
                if current_type:
                    self.save_type_json(scene_path, current_type)
                    cv2.destroyAllWindows()
                    return 'next'
                else:
                    print("请先选择类型 (1-5)")
            elif chr(key) in ['1', '2', '3', '4', '5']:
                current_type = chr(key)
                type_name = self.type_mapping[current_type]
                print(f"已选择类型: {current_type} ({type_name})")
            
            frame_idx += 1
    
    def run(self, start_scene=1, end_scene=50):
        """运行标注工具"""
        print("=" * 60)
        print("场景类型标注工具")
        print("=" * 60)
        print("操作说明:")
        print("  - 按 1-5: 选择场景类型")
        print("  - 按 N: 保存并进入下一个场景")
        print("  - 按 Q: 退出程序")
        print("=" * 60)
        
        scene_num = start_scene
        
        while scene_num <= end_scene:
            result = self.play_scene(scene_num)
            
            if result == 'quit':
                print("\n标注已退出")
                break
            elif result == 'next':
                scene_num += 1
            elif result is None:
                # 场景不存在，跳到下一个
                scene_num += 1
        
        print("\n标注完成!")


if __name__ == "__main__":
    # ============ 配置参数 ============
    # 根据你的实际路径修改以下参数
    
    # 示例1: /data/jingjing/data/context/realdata_sampled_20251110/train/task_0103_user_0555_scene_0001_cfg_0001/cam_104122063550
    base_path = "/data/jingjing/data/context/realdata_sampled_20251111/train"
    task_id = "task_0103"
    user_id = "user_0555"
    cam_id = "cam_104122063550"
    
    # 示例2: 如果是不同的日期或路径
    # base_path = "/data/jingjing/data/context/realdata_sampled_20251030/train"
    # task_id = "task_0102"
    # user_id = "user_0555"
    # cam_id = "cam_104122063550"
    
    print("=" * 60)
    print("配置信息:")
    print(f"  基础路径: {base_path}")
    print(f"  Task ID: {task_id}")
    print(f"  User ID: {user_id}")
    print(f"  Camera ID: {cam_id}")
    print("=" * 60)
    
    # 创建标注工具
    tool = SceneAnnotationTool(base_path, task_id, user_id, cam_id)
    
    # 运行标注，从scene_0001到scene_0050
    tool.run(start_scene=51, end_scene=99)