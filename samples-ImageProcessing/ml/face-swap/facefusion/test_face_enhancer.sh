#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

mkdir -p content

detect_model="yunet_2023mar" # retinaface_10g or yunet_2023mar
recognizer_model="arcface_w600k_r50" # arcface_w600k_r50 or arcface_simswap

enh_model="gfpgan_1.4"

for enh_model in models/face-enh/*.onnx; do
  enh_model=${enh_model##*/}
  enh_model=${enh_model%.onnx}
  echo "Invoking face-enhancement model '${enh_model}' ..."

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
      face-enh \
      -i /content/src.png \
      -o /content/result-enh-faces-${enh_model}.png \
      --detect-model $detect_model \
      --rec-model $recognizer_model \
      --face-enh-model $enh_model
done

