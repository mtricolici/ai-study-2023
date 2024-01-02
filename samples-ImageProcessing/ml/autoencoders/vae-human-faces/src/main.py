#!/usr/bin/env/python
import argparse
import tensorflow as tf
import numpy as np

from model import VAE
from image import save_image, save_images_as_grid
from helper import lm

#########################################################
def invoke_train():
    lm('creating model ...')
    vae = VAE()
    lm('starting training ...')
    try:
      vae.train()
    except KeyboardInterrupt:
        lm('\nAborting ...')
#########################################################
def invoke_demo():
    vae = VAE()
    vae.load_model()
    samples = vae.generate_samples(20)
    save_images_as_grid(samples, '/content/grid.jpg', 5)
#########################################################
def show_info():
    if tf.config.list_physical_devices('GPU'):
        lm("Available GPUs:")
        for gpu in tf.config.list_physical_devices('GPU'):
            lm(gpu)
    else:
        lm("GPU is not available :((")
#########################################################
def main():
    lm(f"TensorFlowVersion: {tf.__version__}")

    parser = argparse.ArgumentParser(description='GAN try 2')
    parser.add_argument('command', choices=['train', 'demo', 'info'], help='The command to execute')
    args = parser.parse_args()

    if args.command == 'train':
        invoke_train()
    elif args.command == 'demo':
        invoke_demo()
    elif args.command == 'info':
        show_info()

#########################################################
if __name__ == '__main__':
    main()
