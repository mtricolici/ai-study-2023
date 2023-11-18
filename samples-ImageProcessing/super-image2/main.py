#!/usr/bin/env/python
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import cv2

# Define command line arguments
parser = argparse.ArgumentParser(description="Scale up an image 4x using a pre-trained model from TensorFlow Hub.")
parser.add_argument("input_image", type=str, help="Path to the input image")
parser.add_argument("output_image", type=str, help="Path to save the scaled-up image")

args = parser.parse_args()

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"  # ESRGAN model from TensorFlow Hub
model = hub.load(model_url)

# Load and preprocess the input image
input_image = cv2.imread(args.input_image)
input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
input_image = tf.expand_dims(input_image, axis=0)  # Add a batch dimension

# Scale up the image 4x
output_image = model(input_image)

# Save the scaled-up image
output_image = tf.squeeze(output_image, axis=0).numpy()
#output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
cv2.imwrite(args.output_image, output_image)

print(f"Image scaled up 4x and saved to {args.output_image}")

