#!/usr/bin/env/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf

from constants import *
from model import *

#########################################################
def main():
  print(f"TensorFlowVersion: {tf.__version__}")
  parser = argparse.ArgumentParser(description='CNN deblurr ;)')
  parser.add_argument('command', choices=['train', 'continue', 'demo', 'info'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")
    model = model_create()
    train_model(model)
    print("training finished")
    unblure_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)

  elif args.command == 'continue':
    print("Loading existed model from disk ...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    train_model(model)
    print("training finished")
    unblure_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)

  elif args.command == 'demo':
    print("Loading existed model from disk ...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    unblure_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)

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
