#!/bin/bash

IMG=my-nafnet-img-rest

docker run \
  --gpus all \
  -it --rm \
  --network=none \
  -e ORT_LOGGING_LEVEL=ERROR \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.images:/images \
  -v $(pwd)/models:/models:ro \
  -v $(pwd)/.insightface:/home/python/.insightface \
  $IMG \
  python face_detector.py
