#!/usr/bin/env/python
import sys
import argparse
import tensorflow as tf
import numpy as np
import subprocess

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
def invoke_demo_animated(f=0, r=5, s=0.25, d=5):
    vae = VAE()

    if f < 0 or f >= vae.latent_dim:
        print(f'Invalid feature. Allowed range [0 .. {vae.latent_dim-1}]')
        sys.exit(1)

    vae.load_model()

    # Generate 15 random samples
    eps = np.random.normal(size=(15, vae.latent_dim))

    lm(f'Creating content/anim-{f:02d}.gif ...')
    idx = 0

    with open('/tmp/files.txt', 'a') as file:
        for value in np.arange(-r, r, s):
            eps[:, f] = value
            samples = vae.sample(eps)
            jpg=f'/tmp/anim-{idx:03d}.jpg'
            save_images_as_grid(samples, jpg, 5)

            # Add text annotation to image
            txt=f"feature:{f}={value:0.2f}"
            subprocess.run(["bash", "-c",
                f'convert {jpg} -fill white -pointsize 36 -annotate +30+30 "{txt}" {jpg}'])

            # save filename to create GIF later
            file.write(f"{jpg}\n")
            idx += 1

        # write file names in reverse order - smoother animation
        for i in range(idx-1, 0, -1):
            file.write(f"/tmp/anim-{i:03d}.jpg\n")

    # create animated GIF for feature f
    subprocess.run(["bash", "-c",
        f"convert -delay {d} -loop 0 $(< /tmp/files.txt) /content/anim-{f:02d}.gif"])

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

    parser = argparse.ArgumentParser(description='GAN try 2')
    parser.add_argument('command', choices=['train', 'demo', 'demo2', 'info'], help='The command to execute')
    parser.add_argument("-f", "--feature", type=int, default=0, help="feature index (default: 0)")
    parser.add_argument("-r", "--range", type=int, default=5, help="feature range max (default: 5)")
    parser.add_argument("-s", "--step", type=float, default=0.1, help="feature walk step (default: 0.1)")
    parser.add_argument("-d", "--delay", type=int, default=5, help="gif delay between frames (default: 5)")
    args = parser.parse_args()

    if args.command == 'train':
        invoke_train()
    elif args.command == 'demo':
        invoke_demo()
    elif args.command == 'demo2':
        invoke_demo_animated(args.feature, args.range, args.step, args.delay)
    elif args.command == 'info':
        show_info()

#########################################################
if __name__ == '__main__':
    main()
