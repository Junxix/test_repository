#!/bin/bash


BASE_PATH="/data/jingjing/data/context/realdata_sampled_20251030"

for i in {1..50}; do
    SCENE=$(printf "scene_%04d" $i)
    SCENE_PATH="task_0102_user_0555_${SCENE}_cfg_0001"
    
    COLOR_DIR="${BASE_PATH}/train/${SCENE_PATH}/cam_104122063550/color"
    DEPTH_DIR="${BASE_PATH}/train/${SCENE_PATH}/cam_104122063550/depth"
    OUTPUT_DIR="${BASE_PATH}/train/${SCENE_PATH}/cam_104122063550/tapip3d"
    OUTPUT_FILE="${OUTPUT_DIR}/output_data.npz"
    
    echo "processing ${SCENE}..."
    
    mkdir -p "${OUTPUT_DIR}"
    
    python create_tapip3d_npz.py \
        --color_dir "${COLOR_DIR}" \
        --depth_dir "${DEPTH_DIR}" \
        --output "${OUTPUT_FILE}" \
        --depth_scale 1000.0
    echo "---"
done

