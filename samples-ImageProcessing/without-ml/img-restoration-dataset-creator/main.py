#!/usr/bin/env/python
import os
import cv2
import numpy as np
import sys
import random

#######################################################################
def add_grain_effect(image):
  intensity=random.randint(13,17)

  height, width, channels = image.shape

  # Generate random noise with the same dimensions as the image
  noise = np.random.normal(0, intensity, (height, width, channels))

  # Add the noise to the image
  noisy_image = image + noise

  # Ensure that pixel values are within the valid range [0, 255]
  noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
  return noisy_image
#######################################################################
def add_chromatic_aberration(image):
  shift_x=random.randint(1,3)
  shift_y=random.randint(1,3)

  height, width, _ = image.shape
  r_channel = np.roll(image[:, :, 2], shift_x, axis=1)
  b_channel = np.roll(image[:, :, 0], shift_y, axis=0)

  # Combine the shifted channels to create the aberration effect
  output = image.copy()
  output[:, :, 2] = r_channel
  output[:, :, 0] = b_channel

  return output
#######################################################################
def add_scan_lines(image):
  line_thickness=random.randint(1,2)
  scan_line_percentage=random.randint(8,12)
  color_shift_intensity=random.randint(17,23)

  height, width, _ = image.shape
  num_scan_lines = int(height * scan_line_percentage / 100)
  scan_line_step = height // num_scan_lines

  for y in range(0, height, scan_line_step * 2):
    line = image[y:y+line_thickness, :, :]

    # Generate a random color shift
    color_shift = np.random.randint(-color_shift_intensity, color_shift_intensity + 1, 3)
    line_with_color_shift = np.clip(line + color_shift, 0, 255)

    image[y:y+line_thickness, :, :] = line_with_color_shift

  return image
#######################################################################
def add_color_bleeding(image):
  possible_kernel_sizes = [(3, 3), (3, 5), (5, 3), (5, 5)]
  kernel_size = random.choice(possible_kernel_sizes)

  return cv2.GaussianBlur(image, kernel_size, 0)
#######################################################################
def convert(input_file, output_file):
  print(f'>>> {input_file} to {output_file}')
  img = cv2.imread(input_file)

  effects = [add_grain_effect, add_chromatic_aberration, add_scan_lines, add_color_bleeding]

  random.shuffle(effects)
  num_effects_to_apply = random.randint(2, 3)

  for effect in effects[:num_effects_to_apply]:
    img = effect(img)

  cv2.imwrite(output_file, img)
#######################################################################

files = [f for f in os.listdir('/images') if f.endswith("good.png")]
cnt=len(files)
print(f'found {cnt} files')

for i, fn in enumerate(files):
  ofn = os.path.join('/images', fn.replace('good.png', 'bad.png'))
#  if os.path.isfile(ofn):
#    print(f'file {fn} was already converted')
#    continue
  convert(os.path.join('/images', fn), ofn)

print('Done!')
