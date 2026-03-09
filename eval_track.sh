python eval_track.py \
    --data_path /data/jingjing/data/context/realdata_sampled_20260115_mismatch \
    --ckpt_path /data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/mae_adaln_epoch_200_seed_42.ckpt \
    --output_dir eval_embeddings \
    --scene_filter  "task_0107_user_0555_scene_0002_cfg_0001_BEFORE_task_0107_user_0555_scene_0002_cfg_0001_AFTER" \
    --split train \
    --batch_size 1 \
    --num_vis_samples 1 \
    --num_targets 3



