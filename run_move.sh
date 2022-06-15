#!/bin/bash

cd interfacegan

rm -r ../result

python move_a_step.py \
    --save_path ../result \
    --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries/stylegan_celebahq_"$1"/boundary.npy \
    --intercept /home/litianyu/ima206-project-style-gan/interfacegan/boundaries/stylegan_celebahq_"$1"/intercept.npy \
    --latent_space_type Z \
    --num_steps 20 \
    --max_delta 5
    # --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries_ori/stylegan_celebahq_smile_boundary.npy
