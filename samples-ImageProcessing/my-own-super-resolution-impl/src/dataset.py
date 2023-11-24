import os
import numpy as np
import cv2
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def dataset_loader():
  small_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if file.endswith("small.png")]
  big_files = [file.replace("small.png", "big.png") for file in small_files]
  num_files = len(small_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os.exit(1)

  while True:
    indices = np.random.choice(num_files, BATCH_SIZE)
    batch_input = []
    batch_output = []

    for idx in indices:
      batch_input.append(load_image(small_files[idx]))
      batch_output.append(load_image(big_files[idx]))

    batch_x = tf.stack(batch_input)
    batch_y = tf.stack(batch_output)

    yield (batch_x, batch_y)
#########################################################

