#!/bin/bash
set -e

IMG=my-svd

docker build \
    --progress=plain \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG .

