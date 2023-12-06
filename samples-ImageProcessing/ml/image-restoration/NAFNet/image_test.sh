#!/bin/bash

IMG=my-nafnet-img-rest

docker run \
  --gpus all \
  -it --rm \
  --network=none \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.images:/images \
  -v $(pwd)/models:/models:ro \
  $IMG \
  python demo.py \
  -opt /app/NAFNet-width64.yml \
  --input_path /images/src.png \
  --output_path /images/result.png

