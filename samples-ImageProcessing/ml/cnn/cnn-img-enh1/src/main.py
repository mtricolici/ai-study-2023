#!/usr/bin/env/python
import argparse
import tensorflow as tf

from constants import *
from train import *
from helper import psnr_metric

#########################################################
def main():
  print(f"TensorFlowVersion: {tf.__version__}")
#  tf.keras.mixed_precision.set_global_policy('mixed_float16')

  parser = argparse.ArgumentParser(description='CNN deblurr ;)')
  parser.add_argument('command', choices=['train', 'continue', 'demo', 'demo-many', 'info'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")
    model = model_create()
    train_model(model)
    print("training finished")

  elif args.command == 'continue':
    print("Loading existed model from disk ...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'psnr_metric': psnr_metric})
    train_model(model)
    print("training finished")

  elif args.command == 'demo':
    print("Loading existed model from disk ...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'psnr_metric': psnr_metric})
    unblure_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)

  elif args.command == 'demo-many':
    print("Loading existed model from disk ...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'psnr_metric': psnr_metric})
    demo_many(model)

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
