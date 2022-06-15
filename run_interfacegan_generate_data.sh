#!/bin/bash

NUM=$1

cd interfacegan
echo "Generating data..."
python generate_data.py -m stylegan_celebahq -o data/stylegan_celebahq_100k -n "$NUM"