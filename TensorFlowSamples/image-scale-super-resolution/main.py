#!/usr/bin/env/python

import sys

def main():
  if len(sys.argv) != 3:
    print("Usage: 2 arguments required! input and output image path")
    sys.exit(1)
  input_image_path = sys.argv[1]
  output_image_path = sys.argv[2]

  print("Input image:", input_image_path)
  print("Output image:", output_image_path)

if __name__ == "__main__":
  main()
