#!/bin/bash
set -e

IMG=my-human-faces-ds-creator

mkdir -p .insightface # Needed by facedetector to save model here

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
    --progress=plain \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG .
  touch -t 202301011234.56 Dockerfile # BugFix when image is cached - its timestamp never changes
fi

echo "Ready! ;)"
