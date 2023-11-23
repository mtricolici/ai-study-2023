import os
import numpy as np
import cv2
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def dataset_loader():
  ri = np.random.randint(0, RESOLUTIONS_COUNT)
#  print(f">>>dataset_loader called! resolution idx:{ri}")

  # Find all files of the same resolution
  files = [f for f in os.listdir(DATASET_DIR) if os.path.basename(f).startswith(f"{ri}-")]
  good_files = [os.path.join(DATASET_DIR, f) for f in files if GOOD_SUFFIX in f]
  bad_files = [f.replace(GOOD_SUFFIX, BAD_SUFFIX) for f in good_files]
  num_files = len(good_files)
  if num_files == 0:
    print(f"NO Files in dataset with resolution {ri}???")
    os._exit(1)

  indices = np.random.choice(num_files, BATCH_SIZE)
  batch_input = []
  batch_output = []

  for idx in indices:
    batch_input.append(load_image(bad_files[idx]))
    batch_output.append(load_image(good_files[idx]))
#    print(f">>>>dataset: {bad_files[idx]} => {good_files[idx]}")

  batch_x = tf.stack(batch_input)
  batch_y = tf.stack(batch_output)

  return (batch_x, batch_y)
#########################################################

