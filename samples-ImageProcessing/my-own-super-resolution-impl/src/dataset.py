import os
import numpy as np
import cv2
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def load_and_split_dataset():
  small_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if file.endswith("small.png")]
  big_files = [file.replace("small.png", "big.png") for file in small_files]
  num_files = len(small_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os.exit(1)

  split_index = int(num_files * SPLIT_RATIO)

  train_small_files = small_files[:split_index]
  train_big_files = big_files[:split_index]

  validation_small_files = small_files[split_index:]
  validation_big_files = big_files[split_index:]

  return (train_small_files, train_big_files, validation_small_files, validation_big_files)
#########################################################
def calc_validation_steps():
  train_small_files, _, validation_small_files, _ = load_and_split_dataset()
  num_train_samples = len(train_small_files)
  num_validation_samples = len(validation_small_files)

  # round up. for 18.6 will be 19
  validation_steps = int(tf.math.ceil(num_validation_samples / BATCH_SIZE).numpy())

  print("#info:")
  print(f"training-samples   : {num_train_samples}")
  print(f"validation-samples : {num_validation_samples}")
  print(f"batch-size         : {BATCH_SIZE}")
  print(f"validation-steps   : {validation_steps}")

  return validation_steps
#########################################################
def generate_batch(indices, small_files, big_files):
  batch_input = []
  batch_output = []

  for idx in indices:
    batch_input.append(load_image(small_files[idx]))
    batch_output.append(load_image(big_files[idx]))

  batch_x = tf.stack(batch_input)
  batch_y = tf.stack(batch_output)

  return (batch_x, batch_y)
#########################################################
def dataset_loader():
  train_small_files, train_big_files, _, _ = load_and_split_dataset()
  num_train_files = len(train_small_files)

  while True:
    indices = np.random.choice(num_train_files, BATCH_SIZE)
    yield generate_batch(indices, train_small_files, train_big_files)
#########################################################
def validation_dataset_loader():
  _, _, validation_small_files, validation_big_files = load_and_split_dataset()
  num_validation_files = len(validation_small_files)

  while True:
    # replace=False - all indexes should be unique!!! not like in dataset_loader ;)
    indices = np.random.choice(num_validation_files, BATCH_SIZE, replace=False)
    yield generate_batch(indices, validation_small_files, validation_big_files)
#########################################################

