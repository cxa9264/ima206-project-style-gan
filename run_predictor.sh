#!/bin/bash

cd FaceAttrAnalysis

python predict_gender.py \
    `# path to images` \
    --img_path /home/litianyu/ima206-project-style-gan/result \
    `# --img_path /home/litianyu/ima206-project-style-gan/interfacegan/data/stylegan_celebahq_data` \
    `# index of attribute, 20 gender, 31 smile` \
    --target 0 1 2 3 4 5 6 7 8 9 10 11 15 16 20 21 22 31 35 39 \
    --move_target $1 \
    `# save_path, plese make sure the directory exists` \
    --save_path ../move_result \
    --save_plot
    