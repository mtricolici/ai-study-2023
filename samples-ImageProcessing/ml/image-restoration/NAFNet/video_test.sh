#!/bin/bash

IMG=my-nafnet-img-rest

#model="NAFNet-REDS-width64" # <<< deblurr but very aggresive - unreal images :)
#model="NAFNet-SIDD-width64" # << noise removal
#model="NAFNet-GoPro-width64" # << deblur but not very aggresive :)

rm -rf .images/tmp

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
  python main.py \
    -i /images/src.mp4 \
    -o /images/result.mp4 \
    -m NAFNet-REDS-width64 $@
