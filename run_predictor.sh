#!/bin/bash

cd FaceAttrAnalysis

python predict_gender.py \
    `# path to images` \
    --img_path /home/litianyu/ima206-project-style-gan/interfacegan/data/stylegan_celebahq \
    `# index of attribute, 20 gender, 31 smile` \
    --target 20 31 \
    `# save_path, plese make sure the directory exists` \
    --save_path res.csv  
    