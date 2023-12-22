#!/usr/bin/env/python
import argparse
import tensorflow as tf

#########################################################
def invoke_train():
    print('NOT implemented yet')
#########################################################
def invoke_demo():
    print('NOT implemented yet')
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
