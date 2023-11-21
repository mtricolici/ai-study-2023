import tensorflow as tf
from image import load_image, save_image

from constants import *

#########################################################
def split_image(img):
  ih, iw = img.shape[:2]
  tw, th = INPUT_SIZE

  if ih < th or iw < tw:
    raise Exception(f"Minimal input image size expected: {ih}x{iw}")

  if ih == th and iw == tw:
    # no split required ;)
    return [img]

  # Calculate the number of chunks needed
  nx, ny = iw // tw, ih // th
  if iw % tw != 0: nx += 1
  if ih % th != 0: ny += 1

  chunks = []
  for y in range(ny):
    for x in range(nx):
      # Define the boundaries of the chunk
      x1, y1 = x * tw, y * th
      x2, y2 = min(x1 + tw, iw), min(y1 + th, ih)

      # Extract the chunk and pad if necessary
      chunk = img[y1:y2, x1:x2]
      if chunk.shape[0] < th or chunk.shape[1] < tw:
        chunk = tf.image.pad_to_bounding_box(chunk, 0, 0, th, tw)
      chunks.append(chunk)

  return chunks

#########################################################
def scale_image(model, input_path, output_path):
  img = load_image(input_path)

  chunks = split_image(img)
  scaled_chunks = []

  for chunk in chunks:
    chunk = tf.expand_dims(chunk, axis=0) # Add batch dimension
    out = model.predict(chunk)
    out = tf.squeeze(out, axis=0) # Remove batch dimension
    scaled_chunks.append(out)

  for i, sc in enumerate(scaled_chunks):
    save_image(sc, f"/output/chunk-{i+1:02}.png")

#########################################################

