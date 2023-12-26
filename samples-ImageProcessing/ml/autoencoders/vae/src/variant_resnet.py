import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

##################################################################################################################
def _residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut_shape = shortcut.get_shape().as_list()
    if shortcut_shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x
##################################################################################################################
def _sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
##################################################################################################################
def build_encoder(vae):
    inputs = layers.Input(shape=vae.input_shape)
    x = inputs

    for r in vae.rblocks:
        x = _residual_block(x, r)
        x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=layers.LeakyReLU(alpha=vae.relu_alpha))(x)

    z_mean = layers.Dense(vae.latent_dim)(x)
    z_log_var = layers.Dense(vae.latent_dim)(x)
    z = _sampling([z_mean, z_log_var])

    return models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
##################################################################################################################
def build_decoder(vae):
    inputs = tf.keras.Input(shape=(vae.latent_dim,))

    x = layers.Dense(128, layers.LeakyReLU(alpha=vae.relu_alpha))(inputs)

    w = vae.input_shape[0] // 4
    h = vae.input_shape[1] // 4

    x = layers.Dense(w * h * 64, layers.LeakyReLU(alpha=vae.relu_alpha))(x)
    x = layers.Reshape((w, h, 64))(x)

    for r in reversed(vae.rblocks):
        x = _residual_block(x, r)
        x = layers.UpSampling2D()(x)

    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)

    if x.shape[1:] != vae.input_shape:
        raise ValueError(f"Decoder OUTPUT shape {x.shape[1:]} does not match input shape {vae.input_shape}!!!")

    return models.Model(inputs, x, name='decoder')
##################################################################################################################

