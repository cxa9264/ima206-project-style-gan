#!/bin/bash

NUM=$1

cd interfacegan
echo "Generating data..."
python generate_data.py -m stylegan_celebahq -o data/stylegan_celebahq -n "$NUM"

echo "Data generated"
python train_boundary.py \
    -o boundaries/stylegan_celebahq_"$ATTRIBUTE_NAME" \
    -c data/stylegan_celebahq/z.npy \
    -s data/stylegan_celebahq/"$ATTRIBUTE_NAME"_scores.npy