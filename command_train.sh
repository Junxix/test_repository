# example for policy training
torchrun --master_addr 127.0.0.1 --master_port 14527 --nproc_per_node 10 --nnodes 1 --node_rank 0 train.py --data_path /data/jingjing/data/context/realdata_sampled_20260115_mismatch \
    --aug --aug_jitter --num_action 20 --voxel_size 0.005 --obs_feature_dim 512 --hidden_dim 512 --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 --dim_feedforward 2048 --dropout 0.1 \
    --ckpt_dir /data/jingjing/chkpts/su2/rise/task_0107/RISE_relative_track_encoder_preload_cross_attention_weight_semantic_mae_100_no_temp_diff_kv_encoder48 \
    --batch_size 240 --num_epochs 500 --save_epochs 2 --num_workers 32 --seed 233 \
    --track_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/encoder_only_epoch_100_seed_42.ckpt \
    --value_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window48/encoder_human_window48_epoch_50_seed_42.ckpt \
    --value_seq_len 16
# example for data visualization & parameter check
# torchrun --master_addr 192.168.3.50 --master_port 14522 --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py --data_path data/collect_pens --aug --aug_jitter --num_action 20 --voxel_size 0.005 --obs_feature_dim 512 --hidden_dim 512 --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 --dim_feedforward 2048 --dropout 0.1 --ckpt_dir logs/collect_pens --batch_size 1 --num_epochs 1 --save_epochs 1 --num_workers 1 --seed 233 --vis_data
