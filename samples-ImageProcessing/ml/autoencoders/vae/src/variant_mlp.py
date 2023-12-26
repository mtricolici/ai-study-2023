import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

##################################################################################################################
def _reparameterize(z_mean, z_log_var):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

##################################################################################################################
def build_encoder(vae):
    inputs = layers.Input(shape=vae.input_shape)
    x = layers.Flatten()(inputs)

    for d in vae.depths:
        x = layers.Dense(d,
            activation=layers.LeakyReLU(alpha=vae.relu_alpha),
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(vae.l2r))(x)

    z_mean = layers.Dense(vae.latent_dim, kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(vae.l2r))(x)

    z_log_var = layers.Dense(vae.latent_dim, kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(vae.l2r))(x)

    z = _reparameterize(z_mean, z_log_var)
    return models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

##################################################################################################################
def build_decoder(vae):
    inputs = layers.Input(shape=(vae.latent_dim,))

    x = inputs

    for d in reversed(vae.depths):
        x = layers.Dense(d,
            activation=layers.LeakyReLU(alpha=vae.relu_alpha),
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(vae.l2r))(x)

    outputs = layers.Dense(tf.reduce_prod(vae.input_shape), activation='sigmoid',
                           kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(vae.l2r))(x)

    outputs = layers.Reshape(vae.input_shape)(outputs)
    return models.Model(inputs, outputs, name='decoder')
##################################################################################################################

