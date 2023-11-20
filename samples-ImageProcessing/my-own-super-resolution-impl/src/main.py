#!/usr/bin/env/python
import argparse

from model import edsr_model
from train import train


def main():
  parser = argparse.ArgumentParser(description='Super Resolution EDSR demo')
  parser.add_argument('command', choices=['train', 'continue', 'demo'], help='The command to execute')

  args = parser.parse_args()

  if args.command == 'train':
    print("Starting new training...")
    model = edsr_model(num_res_blocks=16, num_filters=64)
    train(model)

  elif args.command == 'continue':
    print("Loading existed model from disk ...")

  elif args.command == 'demo':
    print("let's test super resolution...")


if __name__ == '__main__':
  main()
