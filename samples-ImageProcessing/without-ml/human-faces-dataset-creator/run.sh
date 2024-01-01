#!/bin/bash
set -e

IMG=my-human-faces-ds-creator

mkdir -p .insightface .cache

docker run \
  --gpus all \
  -it --rm \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.insightface:/home/python/.insightface \
  -v $(pwd)/.cache:/home/python/.cache \
  -v $HOME/datasets/raw-images:/raw-images:ro \
  -v $HOME/datasets/human-faces:/human-faces \
  $IMG \
  python main.py \
    --source-path /raw-images \
    --output-path /human-faces \
    --device cuda

