#!/bin/bash

IMG=my-nafnet-img-rest

#model="NAFNet-REDS-width64"    # <<< deblurr but very aggresive - unreal images :)
#model="NAFNet-SIDD-width64"    # << noise removal
#model="NAFNet-GoPro-width64"   # << deblur but not very aggresive :)
#model="Baseline-GoPro-width64" # << ???
#model="Baseline-SIDD-width64"  # << ???

#models=("NAFNet-REDS-width64" "NAFNet-SIDD-width64" "NAFNet-GoPro-width64")
#models=("NAFNet-REDS-width64" "Baseline-GoPro-width64" "Baseline-SIDD-width64")
models=("NAFNet-REDS-width64")

for model in "${models[@]}"; do
  echo "Using model '$model'"
  docker run \
    --gpus all \
    -it --rm \
    --network=none \
    -e PYTHONPATH=/nafnet \
    -w /app \
    -v $(pwd)/src:/app:ro \
    -v $(pwd)/.images:/images \
    -v $(pwd)/models:/models:ro \
    $IMG \
    python demo.py \
    -opt /app/$model.yml \
    --input_path /images/src.png \
    --output_path /images/result-${model}.png
done


