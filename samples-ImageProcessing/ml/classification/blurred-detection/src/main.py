#!/usr/bin/env/python
import argparse
import tensorflow as tf

import train as t
import model as m

#########################################################
def main():
    print(f"TensorFlowVersion: {tf.__version__}")

    parser = argparse.ArgumentParser(description='Blur detection')
    parser.add_argument('command', choices=['train', 'demo', 'info'], help='The command to execute')

    args = parser.parse_args()

    if args.command == 'train':
      print("Starting new training...")
      model = m.create_model()
      t.train_model(model)
      print("training finished")

    elif args.command == 'demo':
      print('not implemented yet')

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


