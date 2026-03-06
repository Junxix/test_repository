import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageViewerWithTimestamp:
    def __init__(self, base_dir):
        # 初始化路径和变量
        self.base_dir = base_dir
        self.directory_path = os.path.join(base_dir, "cam_104122063550/color")
        self.current_index = 0
        self.timestamps = {
            'start_time1': '',
            'finish_time1': '',
            'start_time2': '',
            'finish_time2': '',
            'robor_start_time': ''
        }
        
        # 加载PNG文件列表
        self.load_png_files()
        
        # 设置matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 显示第一张图片
        self.display_current_image()
        
    def load_png_files(self):
        """加载并排序PNG文件"""
        if not os.path.exists(self.directory_path):
            raise ValueError(f"目录不存在: {self.directory_path}")
            
        self.png_files = [f for f in os.listdir(self.directory_path) if f.endswith('.png')]
        self.png_files.sort()  # 按文件名排序（假设文件名包含时间戳）
        
        if not self.png_files:
            raise ValueError(f"在目录 {self.directory_path} 中没有找到PNG文件")
            
        print(f"找到 {len(self.png_files)} 个PNG文件")
        
    def display_current_image(self):
        """显示当前图片和状态信息"""
        self.ax.clear()
        
        if 0 <= self.current_index < len(self.png_files):
            # 加载并显示图片
            current_file = self.png_files[self.current_index]
            image_path = os.path.join(self.directory_path, current_file)
            
            try:
                img = Image.open(image_path)
                self.ax.imshow(img)
                self.ax.axis('off')  # 隐藏坐标轴
                
                # 显示当前文件信息
                current_timestamp = current_file.split('.')[0]
                title = f"图片 {self.current_index + 1}/{len(self.png_files)}: {current_file}"
                self.ax.set_title(title, fontsize=12, pad=20)
                
                # 在图片上添加状态信息
                self.add_status_overlay(current_timestamp)
                
            except Exception as e:
                self.ax.text(0.5, 0.5, f"无法加载图片: {str(e)}", 
                           transform=self.ax.transAxes, ha='center', va='center')

        plt.draw()
        
    def add_status_overlay(self, current_timestamp):
        """在图片上添加状态覆盖层"""
        # 创建半透明背景
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
        
        # 显示当前时间戳
        status_text = f"当前时间戳: {current_timestamp}\n\n"
        
        # 显示已记录的时间戳
        status_text += "已记录的时间戳:\n"
        for key, value in self.timestamps.items():
            marker = "✓" if value else "○"
            status_text += f"{marker} {key}: {value}\n"
        
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', bbox=bbox_props, color='white')

        
    def on_key_press(self, event):
        """处理键盘事件"""
        if event.key == 'p':  # 上一张图片
            if self.current_index > 0:
                self.current_index -= 1
                self.display_current_image()
                print(f"切换到图片 {self.current_index + 1}")
            else:
                print("已经是第一张图片")
                
        elif event.key == 'n':  # 下一张图片
            if self.current_index < len(self.png_files) - 1:
                self.current_index += 1
                self.display_current_image()
                print(f"切换到图片 {self.current_index + 1}")
            else:
                print("已经是最后一张图片")
                
        elif event.key in ['1', '2', '3', '4']:  # 记录时间戳
            current_timestamp = self.png_files[self.current_index].split('.')[0]
            
            if event.key == '1':
                self.timestamps['start_time1'] = current_timestamp
                print(f"记录 start_time1: {current_timestamp}")
            elif event.key == '2':
                self.timestamps['finish_time1'] = current_timestamp
                print(f"记录 finish_time1: {current_timestamp}")
            elif event.key == '3':
                self.timestamps['start_time2'] = current_timestamp
                print(f"记录 start_time2: {current_timestamp}")
            elif event.key == '4':
                self.timestamps['finish_time2'] = current_timestamp
                self.timestamps['robor_start_time'] = current_timestamp  # 同时设置robor_start_time
                print(f"记录 finish_time2 和 robor_start_time: {current_timestamp}")
                
            self.display_current_image()  # 刷新显示
            
        # elif event.key == 's':  # 保存时间戳
        #     self.save_timestamps()
            
        elif event.key == 'q':  # 退出
            self.save_timestamps()
            print("退出程序")
            plt.close('all')
            
    def save_timestamps(self):
        """保存时间戳到JSON文件"""
        output_file = os.path.join(self.base_dir, 'human.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.timestamps, f, indent=2, ensure_ascii=False)
            print(f"时间戳已保存到: {output_file}")
            print("保存的时间戳:", self.timestamps)
        except Exception as e:
            print(f"保存时间戳时出错: {str(e)}")
            
    def run(self):
        """运行图片查看器"""
        print("图片查看器启动！")
        print("使用 p/n 切换图片，1-4 记录时间戳，s 保存，q 退出")
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 请替换为您的实际基础目录路径
    base_dir = "/home/ubuntu/data/realdata_20260202_val/task_0108_user_0555_scene_0004_cfg_0001"

    viewer = ImageViewerWithTimestamp(base_dir)
    viewer.run()
