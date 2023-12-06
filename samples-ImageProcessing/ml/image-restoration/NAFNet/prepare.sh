#!/bin/bash

IMG=my-nafnet-img-rest

if [ ! -f "nafnet/requirements.txt" ]; then
  git clone https://github.com/megvii-research/NAFNet.git nafnet
  rm -rf nafnet/.git # Need to remove this otherwise parent .gitignore does not work :(
fi

if [ ! -f "models/NAFNet-REDS-width64.pth" ]; then
  echo "NAFNet-REDS-width64.pth not found!"
  echo "Download it from: https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models"
  exit 1
fi

mkdir -p .images # Put your images here

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
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG .
  touch -t 202301011234.56 Dockerfile # BugFix when image is cached - its timestamp never changes
fi

echo "Ready! ;)"
