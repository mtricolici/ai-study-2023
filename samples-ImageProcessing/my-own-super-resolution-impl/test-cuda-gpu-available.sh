#!/bin/bash

docker run --gpus all --rm tensorflow/tensorflow:latest-gpu python -c "$(cat << EOF
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  print("Available GPUs:")
  for gpu in gpus:
      print(gpu)
  EOF
else:
  print("GPU is not available :((")
EOF
)"
