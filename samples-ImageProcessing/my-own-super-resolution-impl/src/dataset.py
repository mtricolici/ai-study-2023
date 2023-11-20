import os
import numpy as np
import cv2
import tensorflow as tf

from constants import *

#########################################################
def load_image(path):
#  img = cv2.imread(path)
#  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = tf.io.read_file(path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img

#########################################################
def dataset_loader():
  small_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if SMALL_SUFFIX in file]
  big_files = [file.replace(SMALL_SUFFIX, BIG_SUFFIX) for file in small_files]
  num_files = len(small_files)

  while True:
    indices = np.random.choice(num_files, BATCH_SIZE)
    batch_input = []
    batch_output = []

    for idx in indices:
      batch_input.append(load_image(small_files[idx]))
      batch_output.append(load_image(big_files[idx]))

    #batch_x = np.array(batch_input, dtype='float32') / 255.0
    #batch_y = np.array(batch_output, dtype='float32') / 255.0
    batch_x = tf.stack(batch_input)
    batch_y = tf.stack(batch_output)

    yield (batch_x, batch_y)
#########################################################

