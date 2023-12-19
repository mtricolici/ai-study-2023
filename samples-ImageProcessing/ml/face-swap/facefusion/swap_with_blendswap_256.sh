#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

mkdir -p content

detect_model="retinaface_10g"
rec_model="arcface_simswap" # or arcface_w600k_r50 - I do not see any difference
swap_model="blendswap_256"

rm -f content/result-blendswap256.png

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
    -o /content/result-blendswap256.png \
    -f /content/face.png \
    --detect-model $detect_model \
    --swap-model $swap_model \
    --rec-model $rec_model
