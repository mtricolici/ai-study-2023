#!/bin/bash
set -e

IMG=my-facefusion

function check_model() {
  local mf="$1"

  if [ ! -f "models/${mf}" ]; then
    echo "Error: file 'models/${mf}' not found!"
    echo "Download it from: https://github.com/facefusion/facefusion-assets"
    echo "or invoke ./download_models.sh"
    exit 1
  fi
}

check_model "blendswap_256.onnx"
check_model "inswapper_128.onnx"
check_model "inswapper_128_fp16.onnx"
check_model "simswap_256.onnx"
check_model "simswap_512_unofficial.onnx"

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
