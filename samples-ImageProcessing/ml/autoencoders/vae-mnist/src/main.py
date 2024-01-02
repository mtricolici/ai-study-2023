#!/usr/bin/env/python

import argparse
import tensorflow as tf
import numpy as np

from helper import lm, save_images_as_grid
from model import CVAE
from train import train

#########################################################
def invoke_train():
    lm('creating model ...')
    model = CVAE()
    lm('start training ...')
    train(model)
    lm('training finished')
#########################################################
def invoke_demo(num_samples=20, items_per_row=5):
    lm('loading model ...')
    model = CVAE()
    model.load_model()
    lm('generating samples ...')

    eps = np.random.normal(size=(num_samples, model.latent_dim))
    samples = model.sample(eps)
    save_images_as_grid(samples, '/content/grid.jpg', items_per_row)
    lm('Done!')
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

    parser = argparse.ArgumentParser(description='VAE mnist')
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
