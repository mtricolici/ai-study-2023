#!/bin/bash

IMG=my-nafnet-img-rest

docker run \
  --gpus all \
  -it --rm \
  -u 0 \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.images:/images \
  -v $(pwd)/models:/models:ro \
  $IMG \
  bash
