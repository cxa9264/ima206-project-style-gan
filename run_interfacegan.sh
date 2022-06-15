#!/bin/bash

INPUT_CODE="/home/litianyu/ima206-project-style-gan/interfacegan/data/stylegan_celebahq_data/z.npy"
INPUT_SCORE="/home/litianyu/ima206-project-style-gan/svm_data/"$1".npy"
OUTPUT="boundaries"
ATTRIBUTE_NAME=$2

echo "Training boundary..."
cd interfacegan

python train_boundary.py \
    -o "$OUTPUT"/stylegan_celebahq_"$ATTRIBUTE_NAME" \
    -c "$INPUT_CODE" \
    -s "$INPUT_SCORE" \
    -n 0.2
