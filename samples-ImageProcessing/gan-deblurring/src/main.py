#!/usr/bin/env/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

from constants import *
from model import MyGanModel


#########################################################
def main():
  print(f"TensorFlowVersion: {tf.__version__}")
  parser = argparse.ArgumentParser(description='GAN deblurr ;)')
  parser.add_argument('command', choices=['train', 'continue', 'demo', 'demo-many', 'info'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")
    model = MyGanModel()
    model.create()
    model.train()
    print("training finished")

  elif args.command == 'continue':
    model = MyGanModel()
    print("Loading existed model from disk ...")
    model.load()
    model.train()
    print("training finished")

  elif args.command == 'demo':
    model = MyGanModel()
    print("Loading existed model from disk ...")
    model.load()
    model.process_image(DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)

  elif args.command == 'demo-many':
    print("TODO: not implemented")

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
