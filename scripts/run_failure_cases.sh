#!/bin/bash

trap "exit" INT TERM

VIDEO_FILENAMES=(
    P01_01_60701_60834_left_pan_4185.mp4
    P01_01_86747_86923_left_pan_0923.mp4
    P01_05_20576_20599_left_pan_1932.mp4
    P01_09_158727_159675_left_plate_3869.mp4
    P01_09_167949_167988_left_plate_0518.mp4
    P01_09_188359_188488_left_bowl_1282.mp4
    P01_09_213936_214009_left_plate_1511.mp4
    P01_14_11953_12048_left_pan_2121.mp4
    P01_14_22972_23226_left_pan_4175.mp4
    P01_14_41843_42036_left_pan_1950.mp4
    P01_14_55908_56170_left_pan_2607.mp4
    P01_14_56189_56272_left_pan_3008.mp4
    P02_03_2499_2585_right_pan_3063.mp4
    P02_03_3192_3239_right_pan_1762.mp4
    P02_03_42849_42906_left_pan_1804.mp4
    P02_03_52485_52565_left_pan_2712.mp4
    P02_03_62865_63263_left_pan_0548.mp4 
    P02_03_67904_68619_right_pan_0326.mp4
    P02_03_69212_71268_left_pan_1042.mp4
    P02_03_71436_71696_left_pan_2999.mp4
    P02_03_73279_74400_left_pan_3322.mp4
    P02_09_102124_102566_right_glass_1480.mp4
    P02_09_84078_84541_left_pan_0950.mp4
    P02_09_87720_87740_left_pan_0992.mp4
    P02_09_87843_89551_left_pan_2198.mp4
    P02_09_91074_91947_left_pan_3254.mp4
    P02_09_92211_92618_left_pan_1683.mp4
    P02_09_93223_93707_left_pan_1564.mp4
)

for VIDEO_FILE in "${VIDEO_FILENAMES[@]}"; do
    echo "Launching Srun for video: ${VIDEO_FILE}"
    srun --gpus=1 python -u demo_run.py -d epic \
        -i /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/stage_two_selected_amt_772_videos \
        -o /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_772videos_wilorspace/2026-02-05_pico_stage3_772videos_wilorspace_maskv2.1_newsf_multiprior_hyper/[template] \
        -l /home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/best_dirs_annotations_772batch.json \
        -s ${VIDEO_FILE}
done