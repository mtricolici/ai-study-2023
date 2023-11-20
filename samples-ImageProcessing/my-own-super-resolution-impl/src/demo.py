import tensorflow as tf
from dataset import load_image

from constants import *

#########################################################
def scale_image(model, input_path, output_path, resize_input=False):
  input_image = load_image(input_path)

  if resize_input:
    input_image = tf.image.resize(input_image, INPUT_SIZE[::-1], method=tf.image.ResizeMethod.BICUBIC)


  # batch dimension is required for predicting a single input!
  input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension

  output = model.predict(input_image)

  # Convert the output tensor to a suitable format for display
  output_image = tf.squeeze(output, axis=0)  # Remove batch dimension
  output_image = tf.clip_by_value(output_image, 0, 1)
  output_image = tf.image.convert_image_dtype(output_image, tf.uint8)
  output_image = tf.image.encode_png(output_image)

  tf.io.write_file(output_path, output_image)
  print(f"Scaled image was saved to {output_path}")

#########################################################

