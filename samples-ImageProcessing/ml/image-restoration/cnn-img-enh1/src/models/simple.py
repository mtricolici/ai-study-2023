import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from constants import *
from helper import psnr_metric

#########################################################
def model_create():
  model = models.Sequential([
    layers.Conv2D(64, (5, 5), padding='same', input_shape=(*INPUT_SIZE[::-1], 3)),
    layers.LeakyReLU(alpha=0.01),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.LeakyReLU(alpha=0.01),
    layers.Conv2DTranspose(3, (3, 3), padding='same')
  ])

  model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mean_squared_error',
    metrics=[psnr_metric])

  model.summary()
  return model

#########################################################
