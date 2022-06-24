#!/bin/bash

cd interfacegan

rm -r ../result

python move_a_step.py \
    --save_path ../result \
    --boundary boundaries/stylegan_celebahq_"$1"_"$2"/boundary.npy \
    --intercept boundaries/stylegan_celebahq_"$1"_"$2"/intercept.npy \
    --latent_space_type "$1" \
    --num_steps 50 \
    --max_delta 5 
    # --on_bound
    # --load_code ../initial_code_z_male_for_manip.npy \
    # --manipulate_boundary \
    #     boundaries/stylegan_celebahq_"$1"_22/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_8/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_5/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_9/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_15/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_24/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_31/boundary.npy \
    #     boundaries/stylegan_celebahq_"$1"_39/boundary.npy \



    # --save_w_code ../initial_code_z_male_for_manip_w.npy \

    # --save_w_code ../initial_code_zw.npy \
    # --save_code ../initial_code_z.npy 

    # --load_code ../initial_code_z.npy \


    # --load_code ../initial_code_z.npy \

    # --load_code ../initial_code_zw.npy \

    # --load_code ../initial_code_zw.npy \


    #     boundaries/stylegan_celebahq_"$1"_33/boundary.npy \
    #     boundaries/stylegan_celebahq_z_8/boundary.npy \
    # --load_code ../initial_code.npy \



    # --boundary /home/litianyu/ima206-project-style-gan/interfacegan/boundaries_ori/stylegan_celebahq_smile_boundary.npy

cd ..
sh run_predictor.sh $2