import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from constants import *

class VAE:
    def __init__(self, latent_dim, input_shape):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None

    def create_model(self):
        # Encoder model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(DEPTH, activation='relu')(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        z = self.reparameterize(z_mean, z_log_var)

        self.encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder model
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(DEPTH, activation='relu')(latent_inputs)

        outputs = tf.keras.layers.Dense(tf.reduce_prod(self.input_shape), activation='sigmoid')(x)
        outputs = tf.keras.layers.Reshape(self.input_shape)(outputs)

        self.decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def save_model(self, model_dir):
        self.encoder.save_weights(model_dir + '/encoder_weights.h5')
        self.decoder.save_weights(model_dir + '/decoder_weights.h5')

    def load_model(self, model_dir):
        self.create_model()
        self.encoder.load_weights(model_dir + '/encoder_weights.h5')
        self.decoder.load_weights(model_dir + '/decoder_weights.h5')


