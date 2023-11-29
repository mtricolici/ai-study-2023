import os
import numpy as np
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def load_and_split_dataset(split_ratio=0.9):
  gray_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if file.endswith("gray.png")]
  color_files = [file.replace("gray.png", "color.png") for file in gray_files]
  num_files = len(gray_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os.exit(1)

  split_index = int(num_files * split_ratio)

  train_gray_files = gray_files[:split_index]
  train_color_files = color_files[:split_index]

  validation_gray_files = gray_files[split_index:]
  validation_color_files = color_files[split_index:]

  return (train_gray_files, train_color_files, validation_gray_files, validation_color_files)
#########################################################
def calc_validation_steps():
  train_gray_files, _, validation_gray_files, _ = load_and_split_dataset()
  num_train_samples = len(train_gray_files)
  num_validation_samples = len(validation_gray_files)

  # round up. for 18.6 will be 19
  validation_steps = int(tf.math.ceil(num_validation_samples / BATCH_SIZE).numpy())

  print("#info:")
  print(f"training-samples   : {num_train_samples}")
  print(f"validation-samples : {num_validation_samples}")
  print(f"batch-size         : {BATCH_SIZE}")
  print(f"validation-steps   : {validation_steps}")

  return validation_steps
#########################################################
def generate_batch(indices, gray_files, color_files):
  batch_input = []
  batch_output = []

  for idx in indices:
    batch_input.append(load_image(gray_files[idx], 1))
    batch_output.append(load_image(color_files[idx], 3))

  batch_x = tf.stack(batch_input)
  batch_y = tf.stack(batch_output)

  return (batch_x, batch_y)
#########################################################
def train_data_loader():
  train_gray_files, train_color_files, _, _ = load_and_split_dataset()
  num_train_files = len(train_gray_files)

  while True:
    indices = np.random.choice(num_train_files, BATCH_SIZE)
    yield generate_batch(indices, train_gray_files, train_color_files)
#########################################################
def validation_data_loader():
  _, _, validation_gray_files, validation_color_files = load_and_split_dataset()
  num_validation_files = len(validation_gray_files)

  while True:
    indices = np.random.choice(num_validation_files, BATCH_SIZE, replace=False)
    yield generate_batch(indices, validation_gray_files, validation_color_files)
#########################################################

