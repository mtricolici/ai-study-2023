import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K

import numpy as np

##################################################################################################################
def build_encoder(vae):
    inputs = tf.keras.Input(shape=vae.input_shape)

    x = layers.Conv2D(vae.f1, 3, 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    for _ in range(2):
        x = layers.Conv2D(vae.f2, 3, 2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(vae.f2, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    mean = layers.Dense(vae.latent_dim, name='mean')(x)
    log_var = layers.Dense(vae.latent_dim, name='log_var')(x)

    return models.Model(inputs, (mean, log_var), name="encoder")
##################################################################################################################
def build_decoder(vae):
    inputs = tf.keras.Input(shape=(vae.latent_dim,))

    x = layers.Dense(vae.u * vae.u * vae.f2)(inputs)
    x = layers.Reshape((vae.u, vae.u, vae.f2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2DTranspose(vae.f2, 3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(vae.f2, 3, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2DTranspose(vae.f1, 3, 2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(3, 3, 1, padding='same', activation='sigmoid')(x)

    if x.shape[1:] != vae.input_shape:
        raise ValueError(f"Decoder OUTPUT shape {x.shape[1:]} does not match input shape {vae.input_shape}!!!")

    return models.Model(inputs, x, name="decoder")
##################################################################################################################
