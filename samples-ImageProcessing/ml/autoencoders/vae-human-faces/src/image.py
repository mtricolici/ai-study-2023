import tensorflow as tf
import numpy as np

#########################################################
def load_image(path):
  img = tf.keras.preprocessing.image.load_img(path)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img / 255.0
  if (img < 0).any() or (img > 1).any():
    raise ValueError(f'Error: Image at "{path}" is not a standard 8-bit per channel image.')
  if np.isnan(img).any():
    raise ValueError(f'Error: Image at "{path}" contains NaN values!?')
  return img

#########################################################
def save_image(img, output_path):
  img = img * 255.0
  img = tf.cast(img, tf.uint8)

  tf.keras.preprocessing.image.save_img(output_path, img)
#########################################################
def save_images_as_grid(images, output_path, items_per_row):
  num_images = len(images)
  num_rows = int(np.ceil(num_images / items_per_row))
  image_height, image_width, _ = images[0].shape
  big_image_height = num_rows * image_height
  big_image_width = items_per_row * image_width
  big_image = np.zeros((big_image_height, big_image_width, 3), dtype=np.uint8)

  for i, img in enumerate(images):
    if isinstance(img, tf.Tensor):
      img = img.numpy()
    row = i // items_per_row
    col = i % items_per_row
    y_start = row * image_height
    y_end = y_start + image_height
    x_start = col * image_width
    x_end = x_start + image_width
    big_image[y_start:y_end, x_start:x_end, :] = (img * 255).astype(np.uint8)

  big_image = tf.convert_to_tensor(big_image)

  tf.keras.preprocessing.image.save_img(output_path, big_image, file_format='jpeg', quality=100)

#########################################################

