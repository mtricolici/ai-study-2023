import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from constants import *
from helper import psnr_metric

#########################################################
def model_create(num_res_blocks=5, num_filters=32):
    # Input Layer
    inputs = layers.Input(shape=(*INPUT_SIZE[::-1], 3))

    # Initial Convolution
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Residual Blocks
    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    # Final Convolution
    x = layers.Conv2D(3, 3, padding='same')(x)

    # Build and Compile
    model = models.Model(inputs=inputs, outputs=x)
    model.compile(
      optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='mean_squared_error',
      metrics=[psnr_metric])

    model.summary()

    return model

#########################################################
def res_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same')(x)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    return layers.add([x, y])
#########################################################
