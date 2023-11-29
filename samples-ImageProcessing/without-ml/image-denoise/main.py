#!/usr/bin/env/python
import cv2
import sys
import numpy as np

def enhance_image(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path)

    # Check if image is loaded
    if image is None:
        print("Error: Image not found.")
        return

    # Increase brightness
    # You can adjust the value of 40 to increase or decrease brightness
    brightness_level = 5

    brightness_matrix = np.ones(image.shape, dtype="uint8") * brightness_level
    brighter_image = cv2.add(image, brightness_matrix)

    # Denoise the image
    # You can adjust the h value to control the strength of denoising
    denoising_strength=2

    denoised_image = cv2.fastNlMeansDenoisingColored(brighter_image, None, 
                                                     h=denoising_strength, 
                                                     hColor=denoising_strength, 
                                                     templateWindowSize=5, 
                                                     searchWindowSize=14)

    # Apply a Gaussian blur for additional smoothing
    smoothed_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)

    # Save the processed image
    cv2.imwrite(output_path, smoothed_image)

    print(f"Enhanced image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        enhance_image(input_path, output_path)

