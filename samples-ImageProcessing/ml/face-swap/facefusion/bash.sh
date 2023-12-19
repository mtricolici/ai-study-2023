#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

docker run \
  --gpus all \
  -u 0 \
  -it --rm \
  --network=none \
  $IMG \
  bash
