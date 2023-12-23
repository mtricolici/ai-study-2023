#!/usr/bin/env/python
import argparse
import tensorflow as tf
import numpy as np

from model import VAE
from image import save_image

latent_dim = 512
input_shape = (128, 128, 3)

#########################################################
def invoke_train():
    print('creating model ...')
    vae = VAE(latent_dim, input_shape)
    print('starting training ...')
    try:
      vae.train()
    except KeyboardInterrupt:
        print('Aborting ...')
    print('saving model ...')
    vae.save_model()
#########################################################
def continue_train():
    print('creating model ...')
    vae = VAE(latent_dim, input_shape)
    vae.load_model()
    print('starting training ...')
    try:
      vae.train()
    except KeyboardInterrupt:
        print('Aborting ...')
    print('saving model ...')
    vae.save_model()
#########################################################
def invoke_demo():
    vae = VAE(latent_dim, input_shape)
    vae.load_model()
    samples = vae.generate_samples(20)
    for i, sample in enumerate(samples):
        filename = f'/content/result-{i + 1}.png'
        save_image(samples[i], filename)
#########################################################
def show_info():
    if tf.config.list_physical_devices('GPU'):
        print("Available GPUs:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(gpu)
    else:
        print("GPU is not available :((")
#########################################################
def main():
    print(f"TensorFlowVersion: {tf.__version__}")

    parser = argparse.ArgumentParser(description='GAN try 2')
    parser.add_argument('command', choices=['train', 'continue', 'demo', 'info'], help='The command to execute')
    args = parser.parse_args()

    if args.command == 'train':
        invoke_train()
    elif args.command == 'continue':
        continue_train()
    elif args.command == 'demo':
        invoke_demo()
    elif args.command == 'info':
        show_info()

#########################################################
if __name__ == '__main__':
    main()
