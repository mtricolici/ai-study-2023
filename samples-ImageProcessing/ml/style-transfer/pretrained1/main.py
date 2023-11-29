#!/usr/bin/env/python
import sys
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Command line arguments
input_image_path_1 = sys.argv[1]
input_image_path_2 = sys.argv[2]
output_image_path = sys.argv[3]

# Load images
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [512, 512])
    return img

img1 = load_img(input_image_path_1)
img2 = load_img(input_image_path_2)

# Load a pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
model = hub.load(model_url)

# Combine images
stylized_image = model(tf.constant(img1[tf.newaxis, ...]), tf.constant(img2[tf.newaxis, ...]))[0]

# Convert the result to image and save
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

result_image = tensor_to_image(stylized_image)
result_image.save(output_image_path)

