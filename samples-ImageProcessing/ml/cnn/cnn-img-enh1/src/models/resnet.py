import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers

from constants import *
from helper import psnr_metric

#########################################################
# num_res_blocks=16, num_filters=64
def model_create(num_res_blocks=14, num_filters=64):
    # Input Layer
    inputs = layers.Input(shape=(None, None, 3))

    # Initial Convolution
    x = layers.Conv2D(num_filters, kernel_size=7, padding='same')(inputs)
    x = layers.Activation('relu')(x)

    # Residual Blocks
    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    # Final Convolution
    x = layers.Conv2D(3, kernel_size=7, padding='same')(x)

    # Build and Compile
    model = models.Model(inputs=inputs, outputs=x)
    model.compile(
      optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='mean_squared_error',
      metrics=[psnr_metric])

#    model.summary()

    return model

#########################################################
def res_block(x, filters, kernel_size=3, stride=1, padding='same'):

    # First convolution layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding)(x)
    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
#    y = layers.LeakyReLU(alpha=0.01)(y)
    y = layers.Activation('relu')(y)

    # Second convolution layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding)(y)
#    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)

    # Add shortcut to the output
    x = layers.Add()([x, y])
#    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Activation('relu')(x)

    return x
#########################################################
