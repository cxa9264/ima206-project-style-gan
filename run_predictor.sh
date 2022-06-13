#!/bin/bash

cd FaceAttrAnalysis

python predict_gender.py \
    --img_path /home/litianyu/ima206-project-style-gan/interfacegan/data/stylegan_celebahq \
    --target 20 3 