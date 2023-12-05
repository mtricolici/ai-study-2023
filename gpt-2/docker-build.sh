#!/bin/bash
set -e

IMG=my-gpt2-img

# seconds since Dockerfile was modified
DOCKERFILE_MODIFIED=$(stat -c %Y Dockerfile)

# check if image exists
if [ -z "$(docker images -q $IMG 2>/dev/null)" ]; then
  IMAGE_CREATED=0
else
  IMAGE_CREATED=$(docker inspect -f '{{.Created}}' $IMG)
  IMAGE_CREATED=$(date -d "$IMAGE_CREATED" +%s)
fi

if [ ! -f "gpt2-src/requirements.txt" ]; then
  mkdir -p gpt2-src
  git clone https://github.com/openai/gpt-2.git gpt2-src
fi

if [ "$DOCKERFILE_MODIFIED" -gt "$IMAGE_CREATED" ]; then
  docker build \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG .
  touch -t 202301011234.56 Dockerfile # When image is cached - its timestamp never changes
fi

