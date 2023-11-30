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
def scale_batch_of_images(model, input_paths, output_paths):
  imgs = [load_image(path) for path in input_paths]
  imgs = tf.stack(imgs)
  scaled = model.predict(imgs, verbose=0)

  for scaled_img, output_path in zip(scaled, output_paths):
    save_image(scaled_img, output_path)

#########################################################
def scale_all(model):
  input_files = [os.path.join('/many-source', f) for f in os.listdir('/many-source') if f.endswith(".png")]
  output_files = [os.path.join('/many-target', os.path.basename(f)) for f in input_files]

  # find files that were converted already and remove from list
  to_remove = []
  for i, output_file in enumerate(output_files):
    if os.path.exists(output_file):
      to_remove.append(i)

  for i in reversed(to_remove):
    del input_files[i]
    del output_files[i]

  nr_of_files = len(input_files)

  for i in range(0, nr_of_files, BATCH_SIZE):
    print(f"scaling batch [{i} .. {i+BATCH_SIZE}] from total {nr_of_files} ...")
    batch_input_files = input_files[i:i + BATCH_SIZE]
    batch_output_files = output_files[i:i + BATCH_SIZE]

    scale_batch_of_images(model, batch_input_files, batch_output_files)

  print("scaling all dataset small files DONE! ;)")

#########################################################

