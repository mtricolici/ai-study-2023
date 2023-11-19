#!/usr/bin/env/python
import sys
import tensorflow as tf
import tensorflow_hub as hub
import cv2

def load_image(image_path):
    """Load an image using OpenCV and convert it to the RGB format."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    """Preprocess the image for the super-resolution model."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)
    return image

def save_image(image, filename):
    """Save an image in PNG format using OpenCV."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

def main(input_path, output_path):
    # Load the image
    image = load_image(input_path)

    # Preprocess the image
    image = preprocess_image(image)

    # Load the pre-trained super-resolution model from TensorFlow Hub
    model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

    # Apply super-resolution
    super_res_image = model(image)

    # Convert the output to a format suitable for saving
    super_res_image = tf.squeeze(super_res_image)
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)
    super_res_image = tf.round(super_res_image)
    super_res_image = tf.cast(super_res_image, tf.uint8)
    super_res_image = super_res_image.numpy()

    # Save the super-resolved image
    save_image(super_res_image, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        main(input_path, output_path)

