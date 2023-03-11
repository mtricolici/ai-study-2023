#!/bin/bash
set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." >/dev/null 2>&1 && pwd)"

docker run --gpus all -it \
  -v $repo_dir:/zzz:ro \
  -v $HOME/temp:/output \
  -w /zzz/ReinforcementLearning/deep-q-learning/python_sample/ \
  --rm \
  -u $(id -u):$(id -g) \
  tensorflow/tensorflow:latest-gpu \
  ./main.py $@
