#!/usr/bin/env/python
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

from model import edsr_model
from train import train
from demo import scale_image, scale_all
from constants import *
from helper import psnr_metric, lm
from isr_model import IsrRdn


#########################################################
def main():
  parser = argparse.ArgumentParser(description='Super Resolution EDSR demo')
  parser.add_argument('command', choices=['train', 'continue', 'scale', 'scale-all', 'info', 'isr'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")
    model = edsr_model()
    train(model)

    print("training finished")

  elif args.command == 'continue':
    lm("Loading existed model from disk ...")

    # safe-mode is needed otherwise it can't deserialize lambda functions :(
    model = load_model(MODEL_SAVE_PATH, safe_mode=False, custom_objects={'psnr_metric': psnr_metric})
    train(model)

    lm("training finished")

  elif args.command == 'scale':
    # safe-mode is needed otherwise it can't deserialize lambda functions :(
    lm("Loading model ...")
    model = load_model(MODEL_SAVE_PATH, safe_mode=False, custom_objects={'psnr_metric': psnr_metric})
    lm("scaling image ...")
    scale_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)
    lm('scaling finished')

  elif args.command == 'scale-all':
    lm("Loading model ...")
    model = load_model(MODEL_SAVE_PATH, safe_mode=False, custom_objects={'psnr_metric': psnr_metric})
    lm("Invoking scale-all")
    scale_all(model)

  elif args.command == 'isr':
    isrModel = IsrRdn('psnr-large') # psnr-large, psnr-small, noise-cancel
    print(f'ISR model loaded: {isrModel} !!!')
    isrModel.model.summary()
    isrModel.model.save(MODEL_SAVE_PATH)
    print(f'ISR model converted to {MODEL_SAVE_PATH} ;)')

  elif args.command == 'info':
    if tf.config.list_physical_devices('GPU'):
      print("Available GPUs:")
      for gpu in tf.config.list_physical_devices('GPU'):
        print(gpu)
    else:
      print("GPU is not available :((")

#########################################################

if __name__ == '__main__':
  main()
