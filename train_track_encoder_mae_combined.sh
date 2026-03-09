#!/bin/bash

# run_train_mae_combined.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --master_port=29501 \
    train_track_encoder_mae_combined.py \
    --data_path /data/jingjing/data/context/realdata_sampled_20260115_mismatch \
    --ckpt_dir /data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window16 \
    --batch_size 240 \
    --num_epochs 200 \
    --lr 1e-4 \
    --mask_ratio 0.5 \
    --mask_strategy random \
    --save_epochs 2 \
    --num_workers 20 \
    --aug \
    --seed 42 \
    --num_targets 3 \
    --num_points 10 \
    --window_size 16 \
    --stride_min 1 \
    --stride_max 30 \
    --sub_batch_size 2560