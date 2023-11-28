import tensorflow as tf

#########################################################
def load_image(path):
  img = tf.keras.preprocessing.image.load_img(path)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img / 255.0
  if (img < 0).any() or (img > 1).any():
    raise ValueError(f'Error: Image at "{path}" is not a standard 8-bit per channel image.')
  return img

#########################################################
def save_image(img, output_path):
  img = img * 255.0
  img = tf.cast(img, tf.uint8)

  tf.keras.preprocessing.image.save_img(output_path, img)
#########################################################
