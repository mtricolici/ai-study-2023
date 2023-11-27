import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from constants import *

#########################################################
def model_create():
  model = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(*INPUT_SIZE[::-1], 3)),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    Conv2DTranspose(3, (3, 3), padding='same')
  ])
  model.compile(optimizer=Adam(LEARNING_RATE), loss='mean_squared_error')
  return model

#########################################################
