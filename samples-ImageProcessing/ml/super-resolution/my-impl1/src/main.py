#!/usr/bin/env/python
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

from model import MyModel
from train import train
from demo import scale_image, scale_all
from constants import *
from helper import *
from isr_model import IsrRdn

#########################################################
def new_training(args):
  lm("Starting new training...")
  model = MyModel().create_model()
  show_model_summary(model)
  train(model)
  lm("training finished")

#########################################################
def continue_training(args):
  lm('Loading existing model from disk into video memory ...')
  model = MyModel().create_model()
  show_model_summary(model)

  lm('Loading weights from disk ...')
  model.load_weights(WEIGHTS_SAVE_PATH)

  lm('Continue training ...')
  train(model)
  lm("training finished")
#########################################################
def scale_one_image(args):
  lm('Loading existing model from disk into video memory ...')
  model = MyModel().create_model()
  show_model_summary(model)
  lm('Loading weights from disk ...')
  model.load_weights(WEIGHTS_SAVE_PATH)
  lm("scaling image ...")
  scale_image(model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)
  lm('scaling finished')

#########################################################
def scale_many_images(args):
  lm('Loading existing model from disk into video memory ...')
  model = MyModel().create_model()
  show_model_summary(model)
  lm('Loading weights from disk ...')
  model.load_weights(WEIGHTS_SAVE_PATH)
  lm("Invoking scale-all")
  scale_all(model)

#########################################################
def test_isr_model(args, model_name):
  isrModel = IsrRdn(model_name) # psnr-large, psnr-small, noise-cancel
  lm(f'ISR model loaded: {isrModel} !!!')
  show_model_summary(isrModel.model)
  scale_image(isrModel.model, DEMO_INPUT_FILE, DEMO_OUTPUT_FILE)
  #isrModel.model.save(MODEL_SAVE_PATH)
  #print(f'ISR model converted to {MODEL_SAVE_PATH} ;)')

#########################################################
def show_gpu_info():
  if tf.config.list_physical_devices('GPU'):
    print("Available GPUs:")
    for gpu in tf.config.list_physical_devices('GPU'):
      print(gpu)
  else:
    print("GPU is not available :((")

#########################################################
def main():
  parser = argparse.ArgumentParser(description='Super Resolution demo')
  parser.add_argument('command', choices=['train', 'continue', 'scale', 'scale-all', 'info', 'isr'], help='The command to execute')
  parser.add_argument('-c', '--cpu', action='store_true', default=False, help='force running in CPU instead of GPU')
  args = parser.parse_args()

  if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  if args.command == 'train':
    new_training(args)

  elif args.command == 'continue':
    continue_training(args)

  elif args.command == 'scale':
    scale_one_image(args)

  elif args.command == 'scale-all':
    scale_many_images(args)

  elif args.command == 'isr':
    test_isr_model(args, 'psnr-large') # psnr-large, psnr-small, noise-cancel

  elif args.command == 'info':
    show_gpu_info()

#########################################################

if __name__ == '__main__':
  main()
