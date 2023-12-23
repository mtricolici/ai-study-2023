import os
import numpy as np
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def generate_batch(indices, files):
  batch = []

  for idx in indices:
    batch.append(load_image(files[idx]))

  return tf.stack(batch)
#########################################################
def data_loader():
  files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if file.endswith(".png")]
  num_files = len(files)

  while True:
    indices = np.random.choice(num_files, BATCH_SIZE)
    yield generate_batch(indices, files)
#########################################################
