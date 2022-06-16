#!/bin/bash

INPUT_CODE="data/stylegan_celebahq_100k/"$1".npy"
INPUT_SCORE="../svm_data_100k/"$2".npy"
OUTPUT="boundaries"
ATTRIBUTE_NAME=$3

echo "Training boundary..."
cd interfacegan

python train_boundary.py \
    -o "$OUTPUT"/stylegan_celebahq_"$1"_"$ATTRIBUTE_NAME" \
    -c "$INPUT_CODE" \
    -s "$INPUT_SCORE" \
    -n 0.02
