import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K

import numpy as np

##################################################################################################################
def _sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
##################################################################################################################
def build_encoder(vae):
    inputs = tf.keras.Input(shape=vae.input_shape)
    x = inputs
    for depth in vae.depths:
        x = layers.Conv2D(depth, 3, activation=layers.LeakyReLU(alpha=vae.relu_alpha),
              strides=2, padding="same", kernel_regularizer=regularizers.l2(vae.l2r))(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    z_mean = layers.Dense(vae.latent_dim, name="z_mean", kernel_regularizer=regularizers.l2(vae.l2r))(x)
    z_log_var = layers.Dense(vae.latent_dim, name="z_log_var", kernel_regularizer=regularizers.l2(vae.l2r))(x)
    z = _sampling([z_mean, z_log_var])
    return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")      
##################################################################################################################
def build_decoder(vae):
    latent_inputs = tf.keras.Input(shape=(vae.latent_dim,))
    x = layers.Dense(vae.latent_space * vae.latent_space * vae.depths[-1], activation=layers.LeakyReLU(alpha=vae.relu_alpha),
                     kernel_regularizer=regularizers.l2(vae.l2r))(latent_inputs)
    x = layers.Reshape((vae.latent_space, vae.latent_space, vae.depths[-1]))(x)

    for depth in reversed(vae.depths):
        x = layers.Conv2DTranspose(depth, 3, activation=layers.LeakyReLU(alpha=vae.relu_alpha),
                strides=2, padding="same", kernel_regularizer=regularizers.l2(vae.l2r))(x)
        x = layers.BatchNormalization()(x)

    outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(vae.l2r))(x)

    if outputs.shape[1:] != vae.input_shape:
        raise ValueError(f"Decoder OUTPUT shape {outputs.shape[1:]} does not match input shape {vae.input_shape}!!!")

    return models.Model(latent_inputs, outputs, name="decoder")
##################################################################################################################

