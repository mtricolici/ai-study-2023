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
def combine_chunks(chunks, original_size):
  oh, ow = original_size
  tw, th = INPUT_SIZE

  # Calculate number of chunks per row and per column
  nx, ny = ow // tw, oh // th
  if ow % tw != 0: nx += 1
  if oh % th != 0: ny += 1

  # Combine chunks row-wise and then column-wise
  rows = []
  for i in range(0, len(chunks), nx):
    row_chunks = chunks[i:i + nx]
    # Make sure all chunks in a row are the same height
    row_height = min(chunk.shape[0] for chunk in row_chunks)
    row_chunks = [chunk[:row_height, :, :] for chunk in row_chunks]
    row = tf.concat(row_chunks, axis=1)
    rows.append(row)

  combined_image = tf.concat(rows, axis=0)

  # Crop the combined image to match the original size
  #combined_image = combined_image[:oh, :ow, :]
  return combined_image

#########################################################

def scale_image(model, input_path, output_path, split=True):
  img = load_image(input_path)
  original_h, original_w = img.shape[:2]

  if split:
    chunks = split_image(img)
    scaled_chunks = []

    for chunk in chunks:
      chunk = tf.expand_dims(chunk, axis=0) # Add batch dimension
      out = model.predict(chunk)
      out = tf.squeeze(out, axis=0) # Remove batch dimension
      scaled_chunks.append(out)

    final_image = combine_chunks(scaled_chunks, (original_h, original_w))
  else:
    if not DYNAMIC_MODEL and (original_w, original_h) != INPUT_SIZE:
      img = tf.image.resize(img, INPUT_SIZE[::-1], method=tf.image.ResizeMethod.LANCZOS3)

    img = tf.expand_dims(img, axis=0) # Add batch dimension
    final_image = model.predict(img)
    final_image = tf.squeeze(final_image, axis=0)  # Remove batch dimension

  save_image(final_image, output_path)

#########################################################

