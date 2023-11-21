#!/bin/bash

#IMG="tensorflow/tensorflow:nightly-gpu"
IMG="tensorflow/tensorflow:latest-gpu"

read -r -d '' python_script <<EOF
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  print("Available GPUs:")
  for gpu in gpus:
      print(gpu)
else:
  print("GPU is not available :((")
EOF

docker run \
  --gpus all \
  --rm $IMG python -c "$python_script"
