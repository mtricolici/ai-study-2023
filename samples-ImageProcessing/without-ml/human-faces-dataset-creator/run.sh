#!/bin/bash
set -e

IMG=my-human-faces-ds-creator

mkdir -p .insightface

docker run \
  --gpus all \
  -it --rm \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.insightface:/home/python/.insightface \
  -v $HOME/temp/raw-images:/raw-images:ro \
  -v $HOME/temp/human-faces-dataset:/human-faces \
  $IMG \
  python main.py \
    --source-path /raw-images \
    --output-path /human-faces \
    --device cuda
