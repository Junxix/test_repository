SCENE_PATH="task_0105_user_0999_scene_0001_cfg_0001"
# 22
python offline_tracking.py \
    --input "/data/jingjing/data/context/realdata_sampled_20260109/train/${SCENE_PATH}/cam_104122063550/tapip3d/output_data.npz" \
    --sam2_checkpoint "/data/jingjing/pretrained-models/sam2/checkpoints/sam2.1_hiera_large.pt" \
    --sam2_config "//data/jingjing/pretrained-models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
    --tapip3d_checkpoint /home/jingjing/workspace/su1/TAPIP3D/checkpoints/tapip3d_final.pth \
    --output_dir /data/jingjing/data/context/realdata_sampled_20260109/train/${SCENE_PATH}/cam_104122063550/sam2_tapip3d_results_offline/ \
    --num_targets_per_segment 2
    # --target_resolution 720 1280

