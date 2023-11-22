import os
import numpy as np
import cv2
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def dataset_loader():
  print(">>>dataset_loader called!")
  good_files = [os.path.join(DATASET_DIR, file) for file in os.listdir(DATASET_DIR) if GOOD_SUFFIX in file]
  bad_files = [file.replace(GOOD_SUFFIX, BAD_SUFFIX) for file in good_files]
  num_files = len(good_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os.exit(1)

  while True:
    indices = np.random.choice(num_files, BATCH_SIZE)
    batch_input = []
    batch_output = []

    for idx in indices:
      batch_input.append(load_image(bad_files[idx]))
      batch_output.append(load_image(good_files[idx]))
#      print(f">>>>dataset: {bad_files[idx]} => {good_files[idx]}")

    #batch_x = tf.stack(batch_input)
    #batch_y = tf.stack(batch_output)

    yield (batch_input, batch_output)
#########################################################

