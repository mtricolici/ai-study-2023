#!/usr/bin/env/python
import argparse
from tensorflow.keras.models import load_model

from model import edsr_model
from train import train
from demo import scale_image
from constants import *


#########################################################
def main():
  parser = argparse.ArgumentParser(description='Super Resolution EDSR demo')
  parser.add_argument('command', choices=['train', 'continue', 'demo'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")

    model = edsr_model(num_res_blocks=16, num_filters=64)
    train(model)

    print("training finished")

  elif args.command == 'continue':
    print("Loading existed model from disk ...")

    model = load_model(MODEL_SAVE_PATH)
    train(model)

    print("training finished")

  elif args.command == 'demo':
    model = load_model(MODEL_SAVE_PATH)
    scale_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE, resize_input=True)

#########################################################

if __name__ == '__main__':
  main()
