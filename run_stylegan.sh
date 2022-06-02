#!/bin/bash

# python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \
#     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

case $1 in
    cifar)
        echo "running with cifar-10"
        cd stylegan2-ada-pytorch
        python generate.py --outdir=out --seeds=0-35 --class=1 \
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl;;

    metfaces)
        echo "running with MetFaces"
        cd stylegan2-ada-pytorch
        python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
            --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl;;
    mixing)
        echo "runing with mixing"
        cd stylegan2-ada-pytorch
        python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
             --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl;;
    *)
        echo "unkown argument";;
esac
