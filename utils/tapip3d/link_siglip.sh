#!/bin/bash

BASE_MISMATCH="/data/jingjing/data/context/realdata_sampled_mismatch_separate_2/train"
BASE_SOURCE="/data/jingjing/data/context/realdata_sampled_20251110/train"

for dir in "$BASE_MISMATCH"/*; do
    [ -d "$dir" ] || continue

    dirname=$(basename "$dir")

    # 解析 BEFORE scene: ...scene_0021_cfg_0001_BEFORE...
    before_scene=$(echo "$dirname" | sed -n 's/.*_scene_\([0-9]\{4\}\)_cfg_0001_BEFORE.*/scene_\1_cfg_0001/p')

    # 解析 AFTER scene: ...scene_0028_cfg_0001_AFTER
    # 注意这里不再写乱七八糟的 BEFORE_task_ 之类了，直接取最后一个 scene_XXXX_cfg_0001_AFTER
    after_scene=$(echo "$dirname" | sed -n 's/.*_scene_\([0-9]\{4\}\)_cfg_0001_AFTER.*/scene_\1_cfg_0001/p')

    echo "----"
    echo "dir:  $dirname"
    echo "BEFORE: $before_scene"
    echo "AFTER : $after_scene"

    cam_dir="$dir/cam_104122063550"
    mkdir -p "$cam_dir"

    # BEFORE -> human_siglip
    if [ -n "$before_scene" ]; then
        src_human="$BASE_SOURCE/task_0103_user_0555_${before_scene}/cam_104122063550/human_siglip"
        if [ -d "$src_human" ]; then
            ln -snf "$src_human" "$cam_dir/"
            echo "✅ linked human_siglip -> $src_human"
        else
            echo "❌ missing human_siglip dir: $src_human"
        fi
    else
        echo "⚠ 未解析出 BEFORE scene"
    fi

    # AFTER -> robot_siglip
    if [ -n "$after_scene" ]; then
        src_robot="$BASE_SOURCE/task_0103_user_0555_${after_scene}/cam_104122063550/robot_siglip"
        if [ -d "$src_robot" ]; then
            ln -snf "$src_robot" "$cam_dir/"
            echo "✅ linked robot_siglip -> $src_robot"
        else
            echo "❌ missing robot_siglip dir: $src_robot"
        fi
    else
        echo "⚠ 未解析出 AFTER scene"
    fi

done
