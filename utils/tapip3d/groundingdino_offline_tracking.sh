#!/bin/bash

# --- 固定的基本路径和参数 ---
# (您可以根据需要修改这些)

# 场景编号的前缀和后缀
SCENE_PREFIX="task_0103_user_0555"
SCENE_SUFFIX="_cfg_0001"

# 固定的checkpoint路径
SAM2_CHECKPOINT="/data/jingjing/pretrained-models/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG="//data/jingjing/pretrained-models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
TAPIP3D_CHECKPOINT="/home/jingjing/workspace/su1/TAPIP3D/checkpoints/tapip3d_final.pth"
DINO_CONFIG="/data/jingjing/pretrained-models/groundingdino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT="/data/jingjing/pretrained-models/groundingdino/groundingdino_swint_ogc.pth"

# 固定的检测参数
TEXT_PROMPT="yellow object . green object."
BOX_THRESHOLD=0.25
TEXT_THRESHOLD=0.25

# 基础数据路径
BASE_DATA_PATH="/data/jingjing/data/context/realdata_sampled_20251107_tapip3d/train"
BASE_OUTPUT_PATH="/data/jingjing/data/context/realdata_sampled_20251107/train"
CAMERA_ID="cam_104122063550"


# --- 循环执行 ---
# 循环遍历从 21 到 50 的所有数字
for i in $(seq 21 50); do
    # 1. 将数字格式化为4位数（例如: 21 -> 0021）
    SCENE_NUM=$(printf "%04d" $i)
    
    # 2. 构建完整的 SCENE_PATH 变量
    SCENE_PATH="${SCENE_PREFIX}_scene_${SCENE_NUM}${SCENE_SUFFIX}"
    
    # 3. 构建动态的输入和输出路径
    INPUT_FILE="${BASE_DATA_PATH}/${SCENE_PATH}/${CAMERA_ID}/tapip3d/output_data.npz"
    OUTPUT_DIR="${BASE_OUTPUT_PATH}/${SCENE_PATH}/${CAMERA_ID}/groundingdino_sam2_tapip3d_results_offline/"
    
    # --- 打印信息并执行命令 ---
    echo ""
    echo "======================================================================"
    echo " processing: ${SCENE_PATH}"
    echo "      Input: ${INPUT_FILE}"
    echo "     Output: ${OUTPUT_DIR}"
    echo "======================================================================"
    echo ""
    
    # 确保输出目录存在 (如果脚本需要)
    mkdir -p "${OUTPUT_DIR}"
    
    # 4. 执行Python脚本
    python groundingdino_offline_tracking.py \
        --input "${INPUT_FILE}" \
        --sam2_checkpoint "${SAM2_CHECKPOINT}" \
        --sam2_config "${SAM2_CONFIG}" \
        --tapip3d_checkpoint "${TAPIP3D_CHECKPOINT}" \
        --grounding_dino_config "${DINO_CONFIG}" \
        --grounding_dino_checkpoint "${DINO_CHECKPOINT}" \
        --text_prompt "${TEXT_PROMPT}" \
        --box_threshold ${BOX_THRESHOLD} \
        --text_threshold ${TEXT_THRESHOLD} \
        --output_dir "${OUTPUT_DIR}" \
        --visualize \
        --erode_mask

done

echo ""
echo "--- 批量处理完成 (scene_0021 到 scene_0050) ---"