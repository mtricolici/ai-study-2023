import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers

from constants import *
from helper import psnr_metric

#########################################################
# num_res_blocks=16, num_filters=64
def model_create(num_res_blocks=32, num_filters=128):
    # Input Layer
    inputs = layers.Input(shape=(None, None, 3))

    # Initial Convolution
    x = layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual Blocks
    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    # Final Convolution
    x = layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    # Build and Compile
    model = models.Model(inputs=inputs, outputs=x)
    model.compile(
      optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
      loss='mean_squared_error',
      metrics=[psnr_metric])

    model.summary()

    return model

#########################################################
def res_block(input_tensor, filters, kernel_size=(3,3), strides=(1,1), padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

#    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
#    y = layers.LeakyReLU(alpha=0.01)(y)
#    y = layers.Activation('relu')(y)

#########################################################
