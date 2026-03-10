python eval_offline.py \
    --data_path /data/jingjing/data/context/realdata_sampled_20260202_val/ \
    --ckpt /data/jingjing/chkpts/su2/rise/task_0107/task_0107_HistRISE_relative_track_encoder_weight_semantic_cross_attention_preload_mae_50_window16/policy_epoch_400_seed_233.ckpt \
    --num_targets 3 \
    --max_test_steps 200 \
    --vis  \
    --scene_filter  "task_0108_user_0555_scene_0003_cfg_0001_BEFORE_task_0108_user_0555_scene_0003_cfg_0001_AFTER" \
    --save_results