import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
from constants import *
from helper import psnr_metric

#########################################################
def model_create():
  input_layer = layers.Input(shape=(None, None, 1))

  # Encoding layers
  x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.HeNormal())(input_layer)
  x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.HeNormal(), strides=2)(x)
  x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer=initializers.HeNormal())(x)

  # Decoding layers with UpSampling
  x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.HeNormal())(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.HeNormal())(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.HeNormal())(x)

  # Output layer with 3 channels (RGB color)
  output_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', kernel_initializer=initializers.HeNormal())(x)

  model = models.Model(inputs=input_layer, outputs=output_layer)

  model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mean_squared_error',
    metrics=[psnr_metric])

  model.summary()
  return model

#########################################################
