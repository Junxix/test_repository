
"""
批量处理脚本 - 支持 PNG 到 JPG 转换和 SAM2 追踪
处理 scene_0051 到 scene_0099
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import SiglipModel, SiglipImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import glob
import json
import shutil
import cv2

# ==================== 配置 ====================
BASE_DIR = "/data/jingjing/data/context/realdata_sampled_20251110/train"
START_SCENE = 22
END_SCENE = 23
PATCH_SIZE = 14
IMAGE_SIZE = 896
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE

# ==================== 设备设置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 加载模型 ====================
print("=" * 80)
print("加载 SigLIP 模型...")
model_path = os.path.join("/data/pretrained-weights", "siglip-so400m-patch14-384")
model = SiglipModel.from_pretrained(model_path, torch_dtype=torch.float32)
processor = SiglipImageProcessor.from_pretrained(model_path)
vision_model = model.vision_model.to(device)
vision_model.eval()
print(f"✓ SigLIP 模型加载完成！")

# 加载 SAM2
print("\n加载 SAM2 模型...")
USE_SAM2 = False
predictor = None
try:
    from sam2.build_sam import build_sam2_video_predictor
    print("加载 SAM2 模型...")
    sam2_checkpoint = "/data/jingjing/pretrained-models/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "//data/jingjing/pretrained-models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    USE_SAM2 = True
    print("✓ SAM2 模型加载完成！")
except Exception as e:
    print(f"✗ SAM2 加载失败: {e}")
    print("将使用静态 mask（第一帧）")
print("=" * 80)

# ==================== 工具函数 ====================
def denormalize_image(image_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=image_tensor.device, dtype=image_tensor.dtype).view(3, 1, 1)
    return image_tensor * std + mean

def parse_timestamps(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        'robot_start_time': int(data['robor_start_time'])
    }

def classify_images(png_files, robot_start_time):
    human_images, robot_images = [], []
    for png_file in png_files:
        timestamp = int(os.path.splitext(os.path.basename(png_file))[0])
        if timestamp < robot_start_time:
            human_images.append(png_file)
        else:
            robot_images.append(png_file)
    return human_images, robot_images

def convert_png_to_jpg(png_files, tmp_dir, verbose=False):
    """将 PNG 转换为 JPG"""
    os.makedirs(tmp_dir, exist_ok=True)
    png_to_jpg_mapping = {}
    
    iterator = tqdm(png_files, desc="PNG->JPG") if verbose else png_files
    for png_path in iterator:
        basename = os.path.splitext(os.path.basename(png_path))[0]
        jpg_path = os.path.join(tmp_dir, f"{basename}.jpg")
        img = Image.open(png_path).convert('RGB')
        img.save(jpg_path, 'JPEG', quality=95)
        png_to_jpg_mapping[png_path] = jpg_path
    
    return png_to_jpg_mapping

def extract_siglip_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image_normalized = (image_array - mean) / std
    
    pixel_values = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        features = outputs.last_hidden_state
    
    patch_tokens = features.squeeze(0)
    processed_image = denormalize_image(pixel_values[0]).permute(1, 2, 0).cpu().numpy()
    processed_image = np.clip(processed_image, 0, 1)
    
    return patch_tokens, processed_image

def compute_target_features(mask, patch_tokens):
    if isinstance(mask, Image.Image):
        mask_resized = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask_array = np.array(mask_resized)
    else:
        mask_array = cv2.resize(mask.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), 
                               interpolation=cv2.INTER_NEAREST)
    
    mask_binary = (mask_array > 0).astype(np.float32)
    selected_patches, selected_positions = [], []
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y_start, y_end = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
            x_start, x_end = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
            patch_mask = mask_binary[y_start:y_end, x_start:x_end]
            coverage = patch_mask.sum() / (PATCH_SIZE * PATCH_SIZE)
            
            if coverage >= 0.5:
                selected_patches.append(row * GRID_SIZE + col)
                selected_positions.append((row, col))
    
    if len(selected_patches) > 0:
        selected_features = patch_tokens[selected_patches]
        reference_feature = selected_features.mean(dim=0, keepdim=True)
        return reference_feature, selected_positions, mask_binary
    return None, [], mask_binary

def visualize_all_targets(processed_image, targets_data, scene_name, image_name, output_path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 28))
    
    for target_id in range(1, 5):
        row_idx = target_id - 1
        data = targets_data[target_id]
        
        if data['feature'] is not None:
            axes[row_idx, 0].imshow(processed_image)
            axes[row_idx, 0].imshow(data['mask_binary'], alpha=0.5, cmap='Reds')
            for r, c in data['positions']:
                rect = plt.Rectangle((c * PATCH_SIZE, r * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE, 
                                    fill=False, edgecolor='yellow', linewidth=1)
                axes[row_idx, 0].add_patch(rect)
            axes[row_idx, 0].set_title(f'Target {target_id} ({len(data["positions"])} patches) - {image_name}', fontsize=10)
            axes[row_idx, 0].axis('off')
            
            im = axes[row_idx, 1].imshow(data['similarity_map'], cmap='jet', vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f'Target {target_id} Similarity Map', fontsize=10)
            axes[row_idx, 1].axis('off')
            plt.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        else:
            axes[row_idx, 0].imshow(processed_image)
            axes[row_idx, 0].set_title(f'Target {target_id} - No Detection', fontsize=10)
            axes[row_idx, 0].axis('off')
            axes[row_idx, 1].text(0.5, 0.5, 'No Target', ha='center', va='center', fontsize=12)
            axes[row_idx, 1].set_xlim(0, 1)
            axes[row_idx, 1].set_ylim(0, 1)
            axes[row_idx, 1].axis('off')
    
    plt.suptitle(f'{scene_name} - {image_name}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def track_with_sam2(jpg_dir, initial_masks):
    """使用 SAM2 追踪"""
    if not USE_SAM2:
        return None
    
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=jpg_dir)
            
            for target_id, mask in initial_masks.items():
                if mask is not None:
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=target_id,
                        mask=np.array(mask)
                    )
            
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            return video_segments
    except Exception as e:
        print(f"  ✗ SAM2 追踪失败: {e}")
        return None

def process_frames(image_list, video_segments, initial_masks, feature_dict, sim_dir, scene_name, start_idx=0):
    """处理一组帧"""
    for img_idx, image_path in enumerate(tqdm(image_list, desc="Processing", leave=False)):
        image_name = os.path.basename(image_path)
        frame_idx = start_idx + img_idx
        
        patch_tokens, processed_image = extract_siglip_features(image_path)
        
        targets_data = {}
        for target_id in range(1, 5):
            # 获取 mask
            if video_segments and frame_idx in video_segments and target_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][target_id]
            else:
                mask = initial_masks.get(target_id)
            
            if mask is not None:
                feature, positions, mask_binary = compute_target_features(mask, patch_tokens)
                
                if feature is not None:
                    feature_normalized = F.normalize(feature, p=2, dim=1)
                    patch_tokens_normalized = F.normalize(patch_tokens, p=2, dim=1)
                    cosine_sim = torch.mm(feature_normalized, patch_tokens_normalized.T)
                    similarity_map = cosine_sim.squeeze(0).cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
                    
                    targets_data[target_id] = {
                        'feature': feature,
                        'positions': positions,
                        'mask_binary': mask_binary,
                        'similarity_map': similarity_map
                    }
                    feature_dict[target_id].append(feature.cpu().numpy())
                else:
                    targets_data[target_id] = {
                        'feature': None,
                        'positions': [],
                        'mask_binary': mask_binary,
                        'similarity_map': None
                    }
            else:
                targets_data[target_id] = {
                    'feature': None,
                    'positions': [],
                    'mask_binary': np.zeros((IMAGE_SIZE, IMAGE_SIZE)),
                    'similarity_map': None
                }
        
        output_path = os.path.join(sim_dir, f"{image_name.replace('.png', '_similarity.png')}")
        # visualize_all_targets(processed_image, targets_data, scene_name, image_name, output_path)

def process_scene(scene_id):
    """处理单个场景"""
    scene_name = f"scene_{scene_id:04d}"
    scene_base = os.path.join(BASE_DIR, f"task_0103_user_0555_{scene_name}_cfg_0001")
    scene_path = os.path.join(scene_base, "cam_104122063550")
    
    color_dir = os.path.join(scene_path, "color")
    mask_dir = os.path.join(scene_path, "sam2_tapip3d_results_offline")
    json_path = os.path.join(scene_base, "human.json")
    tmp_jpg_dir = os.path.join(scene_path, "tmp_jpg_for_sam2")
    
    # 检查必要文件
    for path in [color_dir, mask_dir, json_path]:
        if not os.path.exists(path):
            return False, f"{os.path.basename(path)} 不存在"
    
    # 创建输出目录
    human_sim_dir = os.path.join(scene_path, "human_siglip_similarity")
    robot_sim_dir = os.path.join(scene_path, "robot_siglip_similarity")
    human_feat_dir = os.path.join(scene_path, "human_siglip")
    robot_feat_dir = os.path.join(scene_path, "robot_siglip")
    
    # for dir_path in [human_sim_dir, robot_sim_dir, human_feat_dir, robot_feat_dir]:
    for dir_path in [human_feat_dir, robot_feat_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # 解析时间戳并分类
    timestamps = parse_timestamps(json_path)
    png_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    if len(png_files) == 0:
        return False, "没有图片"
    
    human_images, robot_images = classify_images(png_files, timestamps['robot_start_time'])
    
    # 加载初始 masks
    initial_masks = {}
    for target_id in range(1, 5):
        mask_path = os.path.join(mask_dir, f"mask_target_{target_id}.png")
        if os.path.exists(mask_path):
            initial_masks[target_id] = Image.open(mask_path).convert('L')
        else:
            initial_masks[target_id] = None
    
    human_features = {i: [] for i in range(1, 5)}
    robot_features = {i: [] for i in range(1, 5)}
    
    try:
        # 转换 PNG 到 JPG
        print(f"  转换 PNG 到 JPG...")
        convert_png_to_jpg(png_files, tmp_jpg_dir)
        
        # SAM2 追踪
        video_segments = None
        if USE_SAM2:
            print(f"  SAM2 追踪...")
            video_segments = track_with_sam2(tmp_jpg_dir, initial_masks)
        
        # 处理 human 和 robot 阶段
        print(f"  处理 Human: {len(human_images)} 帧")
        process_frames(human_images, video_segments, initial_masks, human_features, human_sim_dir, scene_name, 0)
        
        print(f"  处理 Robot: {len(robot_images)} 帧")
        process_frames(robot_images, video_segments, initial_masks, robot_features, robot_sim_dir, scene_name, len(human_images))
        
        # 保存特征
        for target_id in range(1, 5):
            if len(human_features[target_id]) > 0:
                np.save(os.path.join(human_feat_dir, f"target_{target_id}.npy"),
                       np.concatenate(human_features[target_id], axis=0))
            if len(robot_features[target_id]) > 0:
                np.save(os.path.join(robot_feat_dir, f"target_{target_id}.npy"),
                       np.concatenate(robot_features[target_id], axis=0))
        
        return True, f"Human={len(human_images)}, Robot={len(robot_images)}"
    
    finally:
        if os.path.exists(tmp_jpg_dir):
            shutil.rmtree(tmp_jpg_dir)

# ==================== 主程序 ====================
if __name__ == "__main__":
    stats = {'processed': 0, 'failed': 0, 'errors': []}
    
    print(f"\n{'='*80}")
    print(f"批量处理: scene_{START_SCENE:04d} 到 scene_{END_SCENE-1:04d}")
    print(f"{'='*80}\n")
    
    for scene_id in range(START_SCENE, END_SCENE):
        scene_name = f"scene_{scene_id:04d}"
        print(f"\n处理 {scene_name}...")
        
        try:
            success, msg = process_scene(scene_id)
            if success:
                stats['processed'] += 1
                print(f"✓ {scene_name}: {msg}")
            else:
                stats['failed'] += 1
                stats['errors'].append((scene_name, msg))
                print(f"✗ {scene_name}: {msg}")
        except Exception as e:
            stats['failed'] += 1
            error_msg = str(e)
            stats['errors'].append((scene_name, error_msg))
            print(f"✗ {scene_name}: 错误 - {error_msg}")
    
    # 打印统计
    print(f"\n{'='*80}")
    print("处理完成！")
    print(f"  成功: {stats['processed']}")
    print(f"  失败: {stats['failed']}")
    if stats['errors']:
        print("\n失败的场景:")
        for scene, error in stats['errors']:
            print(f"  - {scene}: {error}")
    print(f"{'='*80}")