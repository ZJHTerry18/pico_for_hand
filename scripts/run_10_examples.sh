#!/bin/bash

trap "exit" INT TERM

VIDEO_FILENAMES=(
    P01_01_6059_6109_left_pan_3462.mp4
    P01_01_28797_28850_right_cup_1720.mp4
    P01_01_42298_42980_left_bottle_1106.mp4
    P01_01_57632_58123_left_pan_3813.mp4
    P01_01_83845_84785_left_saucepan_1653.mp4
    P01_01_94721_94774_left_plate_3356.mp4
    P01_05_20843_20935_right_pan_3380.mp4
    P01_09_19563_20121_left_bowl_0151.mp4
    P01_09_94670_94943_left_glass_0786.mp4
    P02_135_12908_12979_right_mug_2643.mp4
)

for VIDEO_FILE in "${VIDEO_FILENAMES[@]}"; do
    echo "Launching Srun for video: ${VIDEO_FILE}"
    srun --gpus=1 python -u demo_run.py -d epic \
        -i /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/stage_two_selected_amt_772_videos \
        -o /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_772videos_wilorspace/2026-02-12_pico_stage3_10s_772videos_wilorspace_maskv3_newsf_multiprior_[template]/ \
        -l /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/best_dirs_annotations_772batch.json \
        -s ${VIDEO_FILE} -r
done