#!/bin/bash
set -e # exit on error

docker build \
  -t pylib_snake \
  .

docker run -it --rm \
  -e DISPLAY="$DISPLAY" \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  pylib_snake \
  /app/main.py
