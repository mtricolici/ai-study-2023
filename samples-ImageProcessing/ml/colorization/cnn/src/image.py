import tensorflow as tf
import numpy as np

#########################################################
def load_image(path, nr_of_colors):
  colors = 'grayscale' if nr_of_colors == 1 else 'rgb'

  img = tf.keras.preprocessing.image.load_img(path, color_mode=colors)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img / 255.0
  if (img < 0).any() or (img > 1).any():
    raise ValueError(f'Error: Image at "{path}" is not a standard 8-bit per channel image.')
  if np.isnan(img).any():
    raise ValueError(f'Error: Image at "{path}" contains NaN values!?')

# uncomment lines bellow to troubleshoot
#  num_channels = img.shape[2] if len(img.shape) == 3 else 1

#  if num_channels == 1:
#    print(f'image {path} - grayscale')
#  else:
#    print(f'image {path} - color RGB')

  return img

#########################################################
def save_image(img, output_path):
  img = img * 255.0
  img = tf.cast(img, tf.uint8)

  tf.keras.preprocessing.image.save_img(output_path, img)
#########################################################
