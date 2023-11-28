import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from constants import *
from helper import psnr_metric

#########################################################
def model_create(depth=8, filters=32):
    inputs = layers.Input(shape=(None, None, 3), name='input')

    # Initial Convolution
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.Activation('relu')(x)

    # Middle convolutional layers
    for _ in range(depth - 2):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = layers.Activation('relu')(x)

    # Final Convolution
    x = layers.Conv2D(3, 3, padding='same', use_bias=False)(x)

    # Build and Compile
    model = models.Model(inputs=inputs, outputs=x)
    model.compile(
      optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='mean_squared_error',
      metrics=[psnr_metric])

    return model
#########################################################
