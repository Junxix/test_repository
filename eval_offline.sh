python eval_offline.py \
    --ckpt /data/jingjing/chkpts/su2/rise/task_0105/RISE_relative_track_encoder_preload_cross_attention_weight_semantic_mae_52_new_ckpt_no_temp/policy_epoch_70_seed_233.ckpt \
    --data_path /data/jingjing/data/context/realdata_sampled_20260107_mismatch \
    --split train \
    --scene_filter  "task_0105_user_0555_scene_0001_cfg_0001_BEFORE_task_0105_user_0555_scene_0001_cfg_0001_AFTER" \
    --track_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/rel_train_all_track_encoder_mae/encoder_only_epoch_100_seed_42.ckpt \
    --value_encoder_ckpt /data/jingjing/chkpts/su2/rise/task_0107/human_track_encoder_mae_window48/encoder_human_window48_epoch_50_seed_42.ckpt \
    --value_seq_len 16 \
    --num_samples 280