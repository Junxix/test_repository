# command_train.sh
torchrun --master_addr 127.0.0.1 --master_port 14528 --nproc_per_node 10 --nnodes 1 --node_rank 0 train.py \
    --data_path /data/jingjing/data/context/realdata_sampled_20260115_mismatch/ \
    --aug --aug_jitter \
    --num_action 20 --num_history 6 \
    --voxel_size 0.005 \
    --obs_feature_dim 512 --hidden_dim 512 \
    --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 --dim_feedforward 2048 --dropout 0.1 \
    --ckpt_dir /data/jingjing/chkpts/su2/rise/task_0107/task_0107_HistRISE_relative_track_encoder_weight_semantic_cross_attention_preload_mae_50_window16 \
    --batch_size 240 --num_epochs 400 --save_epochs 2 --num_workers 32 --seed 233 \
    --track_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/encoder_only_epoch_100_seed_42.ckpt \
    --value_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window16/encoder_human_window16_epoch_50_seed_42.ckpt \
    --value_seq_len 16