#!/bin/bash

cd interfacegan

rm -r ../result

python move_a_step.py \
    --save_path ../result \
    --boundary boundaries/stylegan_celebahq_"$1"_"$2"/boundary.npy \
    --intercept boundaries/stylegan_celebahq_"$1"_"$2"/intercept.npy \
    --latent_space_type "$1" \
    --num_steps 30 \
    --max_delta 2 \
    --load_code ../z_043642.npy \
    # --manipulate_boundary \
    #     boundaries/stylegan_celebahq_z_8/boundary.npy \
    #     boundaries/stylegan_celebahq_z_39/boundary.npy \
    #     boundaries/stylegan_celebahq_z_8/boundary.npy \
    # --save_code ../initial_code.npy


    # --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries_ori/stylegan_celebahq_smile_boundary.npy
