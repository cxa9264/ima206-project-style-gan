#!/bin/bash

NUM=$1

cd interfacegan
echo "Generating data..."
python generate_data.py -m stylegan_celebahq -o data/stylegan_celebahq -n "$NUM"

echo "Data generated"
python train_boundary.py \
    -o boundaries/pggan_celebahq_"$ATTRIBUTE_NAME" \
    -c data/pggan_celebahq/z.npy \
    -s data/pggan_celebahq/"$ATTRIBUTE_NAME"_scores.npy