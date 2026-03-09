#!/bin/bash
# train_human_track_encoder_mae.sh

torchrun --master_addr 127.0.0.1 --master_port 14530 \
    --nproc_per_node 4 --nnodes 1 --node_rank 0 \
    train_human_track_window_encoder_mae.py \
    --data_path /data/jingjing/data/context/realdata_sampled_20260115_mismatch \
    --ckpt_dir /data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window48 \
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
    --window_size 48 \
    --stride_min 1 \
    --stride_max 30 \
    --sub_batch_size 2560