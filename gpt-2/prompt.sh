#!/bin/bash
set -e

IMG=my-gpt2-img
MODEL=774M

#./docker-build.sh

docker run -it --rm \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  -v $(pwd)/gpt2-src:/gpt2 \
  $IMG \
  python src/interactive_conditional_samples.py --model_name $MODEL --top_k 40 --length 256
