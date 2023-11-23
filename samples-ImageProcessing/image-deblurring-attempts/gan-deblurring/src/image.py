import tensorflow as tf

#########################################################
def load_image(path):
  img = tf.keras.preprocessing.image.load_img(path)
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img / 255.0
  return img

#########################################################
def save_image(img, output_path):
  img = img * 255.0
  img = tf.cast(img, tf.uint8)

  tf.keras.preprocessing.image.save_img(output_path, img)
#########################################################
