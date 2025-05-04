#!/bin/bash

IMG=my-svd

mkdir -p content/cache

docker run \
  --gpus all,capabilities=video \
  -it --rm \
  -v $(pwd)/content:/content \
  -v $(pwd)/content/cache:/home/python/.cache \
  -v $(pwd)/src:/app:ro \
  $IMG \
  python /app/main.py
