#!/bin/bash
set -e # exit on error

IMG=my-pylib-snake

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

docker run -it --rm \
  -e DISPLAY="$DISPLAY" \
  -e PYGAME_HIDE_SUPPORT_PROMPT=1 \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  $IMG \
  /app/main.py
