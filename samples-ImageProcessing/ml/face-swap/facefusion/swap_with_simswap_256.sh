#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

mkdir -p content

swap_model="simswap_256"
detect_model="retinaface_10g" # retinaface_10g or yunet_2023mar
rec_model="arcface_simswap" # arcface_simswap or arcface_w600k_r50

rm -f content/result-simswap256.png

docker run \
  --gpus all \
  -it --rm \
  --network=none \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/content:/content \
  -v $(pwd)/models:/models:ro \
  $IMG \
  python /app/main.py \
    swap \
    -i /content/src.png \
    -o /content/result-simswap256.png \
    -f /content/face.png \
    --detect-model $detect_model \
    --swap-model $swap_model \
    --rec-model $rec_model
