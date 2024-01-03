import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K

import numpy as np

##################################################################################################################
def build_encoder(vae):
    inputs = tf.keras.Input(shape=vae.input_shape)
    x = inputs
    for depth in vae.depths:
        x = layers.Conv2D(depth, 3, activation=layers.LeakyReLU(alpha=vae.relu_alpha),
              strides=2, padding="same", kernel_regularizer=regularizers.l2(vae.l2r))(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(vae.latent_dim * 2)(x)

    return models.Model(inputs, x, name="encoder")
##################################################################################################################
def build_decoder(vae):
    inputs = tf.keras.Input(shape=(vae.latent_dim,))

    du = vae.latent_space * vae.latent_space * vae.depths[0]

    x = layers.Dense(units=du, activation=layers.LeakyReLU(alpha=vae.relu_alpha),
                     kernel_regularizer=regularizers.l2(vae.l2r))(inputs)

    x = layers.Reshape((vae.latent_space, vae.latent_space, vae.depths[0]))(x)

    for depth in reversed(vae.depths):
        x = layers.Conv2DTranspose(depth, 3, activation=layers.LeakyReLU(alpha=vae.relu_alpha),
                strides=2, padding="same", kernel_regularizer=regularizers.l2(vae.l2r))(x)
        x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', kernel_regularizer=regularizers.l2(vae.l2r))(x)

    # TODO: not sure if we need this check
    if x.shape[1:] != vae.input_shape:
        raise ValueError(f"Decoder OUTPUT shape {x.shape[1:]} does not match input shape {vae.input_shape}!!!")

    return models.Model(inputs, x, name="decoder")
##################################################################################################################

