#!/bin/bash
set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." >/dev/null 2>&1 && pwd)"

#time docker run --gpus all -it \
time docker run -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $repo_dir:/zzz:ro \
  -v $HOME/temp:/output \
  -w /zzz/ReinforcementLearning/deep-q-learning/python_sample/ \
  --rm \
  mykerasimage \
  ./main.py $@
