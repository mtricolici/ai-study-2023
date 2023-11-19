#!/usr/bin/env/python
import cv2
import numpy as np
import sys
import random

##################################################################
def add_chromatic_aberration(img, shift_x=5, shift_y=5):
    height, width, _ = img.shape
    r_channel = np.roll(img[:, :, 2], shift_x, axis=1)
    b_channel = np.roll(img[:, :, 0], shift_y, axis=0)

    # Combine the shifted channels to create the aberration effect
    output = img.copy()
    output[:, :, 2] = r_channel
    output[:, :, 0] = b_channel

    return output

##################################################################
def add_scan_lines(img, line_thickness=2, scan_line_percentage=10, color_shift_intensity=20):
    height, width, _ = img.shape
    num_scan_lines = int(height * scan_line_percentage / 100)
    scan_line_step = height // num_scan_lines

    for y in range(0, height, scan_line_step * 2):
        line = img[y:y+line_thickness, :, :]

        # Generate a random color shift
        color_shift = np.random.randint(-color_shift_intensity, color_shift_intensity + 1, 3)
        line_with_color_shift = np.clip(line + color_shift, 0, 255)

        img[y:y+line_thickness, :, :] = line_with_color_shift

    return img
##################################################################
def apply_vintage_effect(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply sepia filter
    sepia = np.array([[0.272, 0.534, 0.131],
                      [0.349, 0.686, 0.168],
                      [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia)

    # Apply vignette effect
    height, width, _ = sepia_image.shape
    kernel_x = cv2.getGaussianKernel(width, width * 0.5)
    kernel_y = cv2.getGaussianKernel(height, height * 0.5)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = cv2.filter2D(sepia_image, -1, mask)

    # Combine the sepia and vignette effects
    vintage_image = cv2.addWeighted(sepia_image, 0.7, vignette, 0.3, 0)

    return vintage_image
##################################################################
def add_flicker_effect(image, intensity=0.3):
    flickered_image = image.copy()

    # Generate a random brightness factor for each frame
    brightness_factor = 1.0 + random.uniform(-intensity, intensity)

    # Apply the brightness factor to the image
    flickered_image = cv2.convertScaleAbs(flickered_image, alpha=brightness_factor, beta=0)

    # Clip pixel values to ensure they are within the valid range [0, 255]
    flickered_image = np.clip(flickered_image, 0, 255)

    return flickered_image
##################################################################
def add_gaussian_noise_distortion(image, mean=0, sigma=0.3):
    # Generate Gaussian noise with the specified mean and sigma
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)

    # Add the noise to the input image
    distorted_image = cv2.add(image, noise)

    # Ensure pixel values are within the valid range [0, 255]
    distorted_image = np.clip(distorted_image, 0, 255)

    return distorted_image
##################################################################
# blur - very cpu consuming (slow) !
def add_color_bleeding(image, neighborhood_size=2):
    # Define the size of the neighborhood for color bleeding

    # Create a copy of the input image to work on
    output_image = image.copy()

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Loop through each pixel in the image
    for y in range(neighborhood_size, height - neighborhood_size):
        for x in range(neighborhood_size, width - neighborhood_size):
            # Get the neighborhood around the current pixel
            neighborhood = image[y - neighborhood_size:y + neighborhood_size + 1,
                                x - neighborhood_size:x + neighborhood_size + 1]

            # Calculate the average color of the neighborhood
            average_color = np.mean(neighborhood, axis=(0, 1))

            # Set the color of the current pixel to the average color
            output_image[y, x] = average_color.astype(np.uint8)

    return output_image
##################################################################
def sharpen_image(image, level=8.9):
    # Define a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  level, -1],
                       [-1, -1, -1]])

    # Apply the scaled kernel to the image using convolution
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
##################################################################
def adjust_saturation(image, saturation_factor=0.8):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into its components
    h, s, v = cv2.split(hsv_image)

    # Adjust the saturation component by multiplying it by the saturation factor
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

    # Merge the adjusted components back into the HSV image
    adjusted_hsv_image = cv2.merge((h, s, v))

    # Convert the adjusted HSV image back to BGR color space
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)

    return adjusted_image
##################################################################

def main():
    if len(sys.argv) != 3:
        print("Usage: python vhs_effect.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    try:
        # Read the input image
        img = cv2.imread(input_image_path)

        if img is None:
            print("Error: Unable to read the input image.")
            sys.exit(1)

        # Add various effects to the image
#        img = add_chromatic_aberration(img)
#        img = add_scan_lines(img)
#        img = apply_vintage_effect(img)
#        img = add_flicker_effect(img)
#        img = add_gaussian_noise_distortion(img)

#        img = sharpen_image(img)
#        img = add_color_bleeding(img)
        img = adjust_saturation(img)

        # Save the modified image
        cv2.imwrite(output_image_path, img)

        print(f"Chromatic Aberration effect applied and saved to {output_image_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

