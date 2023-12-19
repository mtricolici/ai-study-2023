#!/bin/bash
set -e

IMG=my-facefusion

if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  echo "Docker image was not built!"
  echo "Invoke ./prepare.sh !"
  exit 1
fi

mkdir -p content

#models=("yunet_2023mar" "retinaface_10g")
models=("yunet_2023mar")
#models=("retinaface_10g")

for model in "${models[@]}"; do
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
      --detect-model $model \
      detect \
      -i /content/src.png \
      -o /content/test-face-detect-$model.png
done


