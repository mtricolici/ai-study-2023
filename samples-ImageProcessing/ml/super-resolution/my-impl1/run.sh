#!/bin/bash
set -e

IMG=my-own-super-resolution

# seconds since Dockerfile was modified
DOCKERFILE_MODIFIED=$(stat -c %Y Dockerfile)

# check if image exists
if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  IMAGE_CREATED=0
else
  IMAGE_CREATED=$(docker inspect -f '{{.Created}}' $IMG)
  IMAGE_CREATED=$(date -d "$IMAGE_CREATED" +%s)
fi

if [ "$DOCKERFILE_MODIFIED" -gt "$IMAGE_CREATED" ]; then
  docker build \
    --no-cache \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG .
fi

mkdir -p output .keras

docker run --gpus all -it --rm \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  --workdir /app \
  -v $(pwd)/src:/app \
  -v $(pwd)/output:/output \
  -v $(pwd)/.keras:/home/python/.keras \
  -v $HOME/temp/many-source:/many-source \
  -v $HOME/temp/many-target:/many-target \
  -v $HOME/temp/super-dataset:/dataset \
  $IMG \
  python /app/main.py $@ || echo "Error: $?"

