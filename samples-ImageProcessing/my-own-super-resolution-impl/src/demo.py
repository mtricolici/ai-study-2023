import os
import tensorflow as tf
from image import load_image, save_image

from constants import *

#########################################################

def scale_image(model, input_path, output_path):
  img = load_image(input_path)
  img = tf.expand_dims(img, axis=0) # Add batch dimension
  final_image = model.predict(img)
  final_image = tf.squeeze(final_image, axis=0)  # Remove batch dimension

  save_image(final_image, output_path)

#########################################################

def scale_all(model, dir_path):
  #TODO: fit model multiple images in batches - should be faster
  files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if '.png' in f]
  for f in files:
    print(f"converting {f}")
    scale_image(model, f, f)

