#!/usr/bin/env/python
import argparse
from ISR.models import RRDN,RDN
from PIL import Image
import numpy as np

def upscale_image(input_path, output_path):
    # Load the model
    #model = RDN(weights='psnr-small')
    model = RDN(weights='psnr-large')
    #model = RDN(weights='noise-cancel')
    #model = RRDN(weights='gans')

    # Load the image
    img = Image.open(input_path)

    # Upscale the image
    sr_img = model.predict(np.array(img))
    sr_img = Image.fromarray(sr_img)

    # Save the upscaled image
    sr_img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale images using ISR")
    parser.add_argument("input_path", help="Path to the input image")
    parser.add_argument("output_path", help="Path to save the output image")

    args = parser.parse_args()

    upscale_image(args.input_path, args.output_path)

