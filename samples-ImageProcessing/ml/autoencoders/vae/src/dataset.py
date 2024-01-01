import os
import numpy as np
import tensorflow as tf

from image import load_image
from helper import lm

#########################################################
def generate_batch(indices, files):
  batch = []

  for idx in indices:
    batch.append(load_image(files[idx]))

  return tf.stack(batch)
#########################################################
def find_files(folder):
  files = []

  for root, _, filenames in os.walk(DATASET_DIR):
    for filename in filenames:
      if filename.endswith(".png"):
        files.append(os.path.join(root, filename))

  return files
#########################################################
def data_loader(batch_size):
  files = find_files('/dataset')
  num_files = len(files)
  lm(f'Dataset - found {num_files} samples.')

  available_indices = np.arange(num_files)
  np.random.shuffle(available_indices)

  while True:
    if len(available_indices) < batch_size:
        available_indices = np.arange(num_files)
        np.random.shuffle(available_indices)

    indices = available_indices[:batch_size]
    available_indices = available_indices[batch_size:]

    yield generate_batch(indices, files)
#########################################################
