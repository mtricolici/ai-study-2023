from tensorflow.keras import optimizers as tf_o
from tensorflow.keras import callbacks as tf_c

from dataset import dataset_loader, validation_dataset_loader, calc_validation_steps
from constants import *
from helper import psnr_metric

#########################################################
def lazy_training(model):
  print('Lazy training !!!')
  for l in model.layers:
    print(f'Layer: {l.__class__}')
  print('---end-of-lazy')
#########################################################
