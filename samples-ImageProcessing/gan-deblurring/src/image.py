import tensorflow as tf

#########################################################
def load_image(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return img

#########################################################
def save_image(output, output_path):
  img = tf.clip_by_value(output, 0, 1)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  img = tf.image.encode_png(img)

  tf.io.write_file(output_path, img)
#########################################################
