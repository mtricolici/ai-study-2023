#!/usr/bin/env/python
import sys
from super_image import AwsrnModel, ImageLoader
from PIL import Image

def super_resolve(input_path, output_path):
    # Load the pre-trained EDSR model
#    model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=4)
#    model = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=4)
#    model = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)
#    model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=4)
#    model = RcanModel.from_pretrained('eugenesiow/rcan-bam', scale=4) << very slow
    model = AwsrnModel.from_pretrained('eugenesiow/awsrn-bam', scale=4)

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

