import os
import numpy as np
import tensorflow as tf

from constants import *
from image import load_image

#########################################################
def dataset_loader():
  good_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith("good.png")]
  bad_files = [f.replace("good.png", "bad.png") for f in good_files]
  num_files = len(good_files)
  if num_files == 0:
    print("NO Files in dataset???")
    os._exit(1)

  while True:
    indices = np.random.choice(num_files, BATCH_SIZE)
    batch_input = []
    batch_output = []

    for idx in indices:
      batch_input.append(load_image(good_files[idx]))
      batch_output.append(load_image(bad_files[idx]))

    batch_x = tf.stack(batch_input)
    batch_y = tf.stack(batch_output)

    yield (batch_x, batch_y)
#########################################################

