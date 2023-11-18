#!/usr/bin/env/python
import sys
from super_image import EdsrModel, ImageLoader
from PIL import Image

def super_resolve(input_path, output_path):
    # Load the pre-trained EDSR model
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    # Open the image using PIL
    image = Image.open(input_path)

    # Load the image for the model
    lr = ImageLoader.load_image(image)

    # Perform super-resolution
    preds = model(lr)

    ImageLoader.save_image(preds, output_path) 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    super_resolve(input_path, output_path)

