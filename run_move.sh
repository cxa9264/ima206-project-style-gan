#!/bin/bash

cd interfacegan

rm -r ../result

python move_a_step.py \
    --save_path ../result \
    --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries_ori/stylegan_celebahq_smile_boundary.npy
    # --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries_ori/stylegan_celebahq_gender_boundary.npy
