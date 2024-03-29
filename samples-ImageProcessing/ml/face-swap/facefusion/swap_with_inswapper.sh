#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

mkdir -p content

recognizer_model="arcface_w600k_r50"
#recognizer_model="arcface_simswap" << NOT GOOD for inswapper!!!

swap_model="inswapper_128"
#swap_model="simswap_256"
#swap_model="simswap_512_unofficial"
#swap_model="blendswap_256"

detect_model="retinaface_10g" # Seems to be the best one

rm -f content/result-inswap.png

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
    -o /content/result-inswap.png \
    -f /content/face.png \
    --detect-model $detect_model \
    --swap-model $swap_model \
    --rec-model $recognizer_model
