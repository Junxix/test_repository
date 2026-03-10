#!/bin/bash

# 批处理脚本：处理scene 0001到0050

# 固定参数
TASK_ID="task_0104"
USER_ID="user_0555"
CFG_ID="cfg_0001"
CAM_ID="cam_104122063550"

BASE_INPUT_DIR="/data/jingjing/data/context/realdata_sampled_20251112_tapip3d/train"
BASE_OUTPUT_DIR="/data/jingjing/data/context/realdata_sampled_20251112/train"

SAM2_CHECKPOINT="/data/jingjing/pretrained-models/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG="//data/jingjing/pretrained-models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
TAPIP3D_CHECKPOINT="/home/jingjing/workspace/su1/TAPIP3D/checkpoints/tapip3d_final.pth"

# 循环处理scene 0001到0050
for i in $(seq 1 50); do
    # 格式化scene编号为4位数字（例如：0001, 0002, ..., 0050）
    SCENE_NUM=$(printf "%04d" $i)
    SCENE_PATH="${TASK_ID}_${USER_ID}_scene_${SCENE_NUM}_${CFG_ID}"
    
    INPUT_PATH="${BASE_INPUT_DIR}/${SCENE_PATH}/${CAM_ID}/tapip3d/output_data.npz"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SCENE_PATH}/${CAM_ID}/yolo_sam2_tapip3d_results/"
    
    echo "========================================"
    echo "处理场景: ${SCENE_PATH}"
    echo "输入: ${INPUT_PATH}"
    echo "输出: ${OUTPUT_DIR}"
    echo "========================================"
    
    # 检查输入文件是否存在
    if [ ! -f "${INPUT_PATH}" ]; then
        echo "警告: 输入文件不存在，跳过: ${INPUT_PATH}"
        echo ""
        continue
    fi
    
    # 运行处理脚本
    python yolo_offline_tracking.py \
        --input "${INPUT_PATH}" \
        --sam2_checkpoint "${SAM2_CHECKPOINT}" \
        --sam2_config "${SAM2_CONFIG}" \
        --tapip3d_checkpoint "${TAPIP3D_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --visualize
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✓ ${SCENE_PATH} 处理成功"
    else
        echo "✗ ${SCENE_PATH} 处理失败"
    fi
    
    echo ""
done

echo "========================================"
echo "全部处理完成！"
echo "========================================"