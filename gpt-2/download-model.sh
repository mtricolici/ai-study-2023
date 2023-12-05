#!/bin/bash
set -e

IMG=my-gpt2-img
MODEL=774M

./docker-build.sh

if [ ! -f "gpt2-src/models/$MODEL/vocab.bpe" ]; then
  echo "Model not found. Downloading it ..."
  docker run -it --rm \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e TF_CPP_MIN_LOG_LEVEL=3 \
    -v $(pwd)/gpt2-src:/gpt2 \
    $IMG \
    python download_model.py $MODEL
else
  echo "Model already present ;) No need to downloaded it again"
fi

