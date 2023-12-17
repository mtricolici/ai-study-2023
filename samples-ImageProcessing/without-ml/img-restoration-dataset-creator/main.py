#!/usr/bin/env/python
import os
import cv2
import numpy as np
import sys
import random

#######################################################################
def add_grain_effect(image):
  intensity=random.randint(10,18)

  height, width, channels = image.shape

  noise = np.random.normal(0, intensity, (height, width, channels))
  noisy_image = image + noise

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
def add_blur_effect(image):
  kernel_size = np.random.choice([3, 5], size=2, replace=True)
  return cv2.GaussianBlur(image, kernel_size, 0)
#######################################################################
def add_downscale_effect(image):
  original_size = image.shape[:2]
  s_min = 0.5
  s_max = 0.7
  scale = s_min + (s_max - s_min)*random.random()
  small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
  return cv2.resize(small, dsize=original_size[::-1], interpolation=cv2.INTER_AREA)
#######################################################################
def add_jpeg_effect(image):
  quality = np.random.randint(40, 70)
  _, jpg_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
  return cv2.imdecode(jpg_image, cv2.IMREAD_UNCHANGED)

#######################################################################

def convert(input_file, output_file):
  print(f'>>> {input_file} to {output_file}')
  img = cv2.imread(input_file)


  effects = [add_downscale_effect, add_blur_effect, add_grain_effect, add_jpeg_effect]
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
