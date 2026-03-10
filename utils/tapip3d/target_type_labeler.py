import os
import json
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Rectangle
import numpy as np

# 类型定义
types = [
    "1: box",
    "2: block", 
    "3: cup",
    "4: doll",
    "5: yellow"
]

# 基础路径
base_path = "/data/jingjing/data/context/realdata_sampled_20251110/train"
task_user_prefix = "task_0103_user_0555"
cfg = "cfg_0001"
cam = "cam_104122063550"

def get_scene_path(scene_num):
    """生成scene路径"""
    scene_dir = f"{task_user_prefix}_scene_{scene_num:04d}_{cfg}"
    return os.path.join(base_path, scene_dir, cam)

class TargetLabeler:
    def __init__(self, scene_path, scene_num):
        self.scene_path = scene_path
        self.scene_num = scene_num
        self.current_target = 1  # 当前正在标注的target
        self.target_types = {}  # 保存标注结果
        self.images = []  # 保存4张图片
        self.fig = None
        self.axes = {}
        self.completed = False
        self.skip = False
        
        # 加载图片
        self.load_images()
        
    def load_images(self):
        """加载4张target图片"""
        self.images = []
        for i in range(1, 5):
            img_path = os.path.join(self.scene_path, "semantic_not_normalized", 
                                   f"dinov3_fully_covered_patches_similarity_target_{i}.png")
            if os.path.exists(img_path):
                img = imread(img_path)
                self.images.append((i, img))
            else:
                self.images.append((i, None))
    
    def create_display(self):
        """创建显示界面"""
        self.fig = plt.figure(figsize=(16, 10))
        scene_dir_name = os.path.basename(os.path.dirname(self.scene_path))
        self.fig.suptitle(f'Scene: {scene_dir_name}', fontsize=16, y=0.98)
        
        # 创建布局：左侧大图，右侧4个小图，底部类型选项和状态
        gs = self.fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], 
                                   height_ratios=[1, 1, 0.15],
                                   hspace=0.3, wspace=0.3)
        
        # 左侧大图（当前target放大显示）
        self.axes['large'] = self.fig.add_subplot(gs[:2, 0])
        self.axes['large'].set_title('', fontsize=14, weight='bold')
        
        # 右侧4个小图
        self.axes['small1'] = self.fig.add_subplot(gs[0, 1])
        self.axes['small2'] = self.fig.add_subplot(gs[0, 2])
        self.axes['small3'] = self.fig.add_subplot(gs[1, 1])
        self.axes['small4'] = self.fig.add_subplot(gs[1, 2])
        
        # 底部类型选项
        self.axes['types'] = self.fig.add_subplot(gs[2, :])
        self.axes['types'].axis('off')
        
        # 显示类型选项
        types_text = "类型选项: " + "  |  ".join(types)
        self.axes['types'].text(0.5, 0.7, types_text, 
                               ha='center', va='center', 
                               fontsize=13, 
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                               weight='bold')
        
        # 显示操作提示
        hint_text = "按 1-5 选择类型  |  按 n 跳到下一个scene  |  按 q 退出"
        self.axes['types'].text(0.5, 0.2, hint_text, 
                               ha='center', va='center', 
                               fontsize=11,
                               style='italic')
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始显示
        self.update_display()
        
        plt.show(block=True)
    
    def update_display(self):
        """更新显示"""
        # 更新左侧大图（当前target）
        self.axes['large'].clear()
        if self.current_target <= 4:
            target_idx = self.current_target - 1
            if self.images[target_idx][1] is not None:
                self.axes['large'].imshow(self.images[target_idx][1])
                
                # 显示当前target信息
                type_str = ""
                if f"target{self.current_target}" in self.target_types:
                    type_num = self.target_types[f"target{self.current_target}"]
                    type_str = f" - 已标注: {types[type_num-1]}"
                
                self.axes['large'].set_title(
                    f'Target {self.current_target}{type_str}', 
                    fontsize=16, weight='bold', color='red'
                )
            else:
                self.axes['large'].text(0.5, 0.5, 'Image not found', 
                                       ha='center', va='center', fontsize=14)
        self.axes['large'].axis('off')
        
        # 更新右侧4个小图
        for i, (target_num, img) in enumerate(self.images):
            ax_name = f'small{i+1}'
            self.axes[ax_name].clear()
            
            if img is not None:
                self.axes[ax_name].imshow(img)
            else:
                self.axes[ax_name].text(0.5, 0.5, 'N/A', 
                                       ha='center', va='center', fontsize=10)
            
            # 标题显示target编号和标注状态
            title = f'T{target_num}'
            color = 'black'
            weight = 'normal'
            
            if f"target{target_num}" in self.target_types:
                type_num = self.target_types[f"target{target_num}"]
                title += f'\n✓ {type_num}'
                color = 'green'
                weight = 'bold'
            
            if target_num == self.current_target:
                # 当前target用红色边框
                self.axes[ax_name].spines['bottom'].set_color('red')
                self.axes[ax_name].spines['top'].set_color('red')
                self.axes[ax_name].spines['left'].set_color('red')
                self.axes[ax_name].spines['right'].set_color('red')
                self.axes[ax_name].spines['bottom'].set_linewidth(4)
                self.axes[ax_name].spines['top'].set_linewidth(4)
                self.axes[ax_name].spines['left'].set_linewidth(4)
                self.axes[ax_name].spines['right'].set_linewidth(4)
            else:
                # 其他target正常边框
                for spine in self.axes[ax_name].spines.values():
                    spine.set_color('gray')
                    spine.set_linewidth(1)
            
            self.axes[ax_name].set_title(title, fontsize=11, color=color, weight=weight)
            self.axes[ax_name].set_xticks([])
            self.axes[ax_name].set_yticks([])
        
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key in ['1', '2', '3', '4', '5']:
            # 为当前target分配类型
            type_num = int(event.key)
            if self.current_target <= 4:
                self.target_types[f"target{self.current_target}"] = type_num
                print(f"Target {self.current_target} -> {types[type_num-1]}")
                
                # 自动进入下一个target
                self.current_target += 1
                
                # 如果完成了所有4个target
                if self.current_target > 4:
                    print("\n✓ 完成当前scene的标注！")
                    print("按 'n' 进入下一个scene，按 'q' 退出")
                    self.completed = True
                
                self.update_display()
        
        elif event.key == 'n':
            # 下一个scene
            if self.completed:
                plt.close(self.fig)
            else:
                print("\n⚠ 当前scene未完成标注，是否确认跳过？")
                self.skip = True
                plt.close(self.fig)
        
        elif event.key == 'q':
            # 退出
            print("\n退出程序")
            plt.close(self.fig)
            exit(0)
        
        elif event.key in ['left', 'right', 'up', 'down']:
            # 方向键切换target
            if event.key == 'right' and self.current_target < 4:
                self.current_target += 1
            elif event.key == 'left' and self.current_target > 1:
                self.current_target -= 1
            self.update_display()

def save_target_types(scene_path, target_types):
    """保存target类型到json文件"""
    sam2_path = os.path.join(scene_path, "sam2_tapip3d_results_offline")
    os.makedirs(sam2_path, exist_ok=True)
    
    json_path = os.path.join(sam2_path, "target_type.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(target_types, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存到: {json_path}")
    print(f"  内容: {json.dumps(target_types, ensure_ascii=False)}")

def main():
    """主函数"""
    scene_num = 21  # 从scene_0021开始
    
    print("="*60)
    print("Target Type 标注工具 (键盘直接输入版)")
    print("="*60)
    print("操作说明:")
    print("  1. 当前target会在左侧放大显示")
    print("  2. 直接按 1-5 键为当前target分配类型（无需回车）")
    print("  3. 分配后自动进入下一个target")
    print("  4. 完成4个target后，按 'n' 进入下一个scene")
    print("  5. 按 'q' 随时退出程序")
    print("  6. 可用方向键 ← → 手动切换target")
    print("="*60)
    
    while scene_num <= 99:
        scene_path = get_scene_path(scene_num)
        
        # 检查路径是否存在
        if not os.path.exists(scene_path):
            print(f"\n⚠ {task_user_prefix}_scene_{scene_num:04d}_{cfg} 路径不存在，跳过...")
            scene_num += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"当前 Scene: {task_user_prefix}_scene_{scene_num:04d}_{cfg}")
        print(f"{'='*60}")
        
        # 创建标注器
        labeler = TargetLabeler(scene_path, scene_num)
        labeler.create_display()
        
        # 保存结果
        if labeler.completed:
            save_target_types(scene_path, labeler.target_types)
            scene_num += 1
        elif labeler.skip:
            print("⚠ 跳过当前scene")
            scene_num += 1
        else:
            # 用户按了q退出
            break
    
    print("\n" + "="*60)
    print("标注完成！")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    finally:
        plt.close('all')