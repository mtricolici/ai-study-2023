import os
import numpy as np
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def load_and_split_dataset(split_ratio=0.8):
  bad_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if file.endswith("bad.png")]
  good_files = [file.replace("bad.png", "good.png") for file in bad_files]
  num_files = len(bad_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os.exit(1)

  split_index = int(num_files * split_ratio)

  train_bad_files = bad_files[:split_index]
  train_good_files = good_files[:split_index]

  validation_bad_files = bad_files[split_index:]
  validation_good_files = good_files[split_index:]

  return (train_bad_files, train_good_files, validation_bad_files, validation_good_files)
#########################################################
def calc_validation_steps():
  train_bad_files, _, validation_bad_files, _ = load_and_split_dataset()
  num_train_samples = len(train_bad_files)
  num_validation_samples = len(validation_bad_files)

  # round up. for 18.6 will be 19
  validation_steps = int(tf.math.ceil(num_validation_samples / BATCH_SIZE).numpy())

  print("#info:")
  print(f"training-samples   : {num_train_samples}")
  print(f"validation-samples : {num_validation_samples}")
  print(f"batch-size         : {BATCH_SIZE}")
  print(f"validation-steps   : {validation_steps}")

  return validation_steps
#########################################################
def generate_batch(indices, bad_files, good_files):
  batch_input = []
  batch_output = []

  for idx in indices:
    batch_input.append(load_image(bad_files[idx]))
    batch_output.append(load_image(good_files[idx]))

  batch_x = tf.stack(batch_input)
  batch_y = tf.stack(batch_output)

  return (batch_x, batch_y)
#########################################################
def train_data_loader():
  train_bad_files, train_good_files, _, _ = load_and_split_dataset()
  num_train_files = len(train_bad_files)

  while True:
    indices = np.random.choice(num_train_files, BATCH_SIZE)
    yield generate_batch(indices, train_bad_files, train_good_files)
#########################################################
def validation_data_loader():
  _, _, validation_bad_files, validation_good_files = load_and_split_dataset()
  num_validation_files = len(validation_bad_files)

  while True:
    indices = np.random.choice(num_validation_files, BATCH_SIZE, replace=False)
    yield generate_batch(indices, validation_bad_files, validation_good_files)
#########################################################

