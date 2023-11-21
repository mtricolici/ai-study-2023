import tensorflow as tf
from image import load_image, save_image

from constants import *

#########################################################
def scale_image(model, input_path, output_path):
  img = load_image(input_path)

  # batch dimension is required for predicting a single input!
  img = tf.expand_dims(img, axis=0)  # Add batch dimension

  output = model.predict(img)

  # Remove batch dimension
  output = tf.squeeze(output, axis=0)

  save_image(output, output_path)

  print(f"Scaled image was saved to {output_path}")

#########################################################

