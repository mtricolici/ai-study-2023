import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from tensorflow.keras import backend as K

import numpy as np

from constants import *
import dataset as ds

class VAE:
#########################################################
    def __init__(self, latent_dim, input_shape, depths):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.depths = depths
        self.create_model()
#########################################################
    def save_model(self):
        self.encoder.save_weights('/content/model_encoder.h5')
        self.decoder.save_weights('/content/model_decoder.h5')
        self.vae.save_weights('/content/model_vae.h5')

#########################################################
    def load_model(self):
        self.encoder.load_weights('/content/model_encoder.h5')
        self.decoder.load_weights('/content/model_decoder.h5')
        self.vae.load_weights('/content/model_vae.h5')
#########################################################
    def generate_samples(self, num_samples):
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(random_latent_points)
#########################################################
    def train(self):
        dl = ds.data_loader()
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))
        self.vae.fit(dl, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCH, verbose=1)
#########################################################
    def create_model(self):
        # Define the encoder
        self.encoder = self.build_encoder()

        # Define the decoder
        self.decoder = self.build_decoder()

        # Combine encoder and decoder to create VAE model
        x = self.encoder.inputs[0]
        z_mean, z_log_var, z = self.encoder(x)
        x_decoded = self.decoder(z)
        self.vae = models.Model(x, x_decoded, name="vae")

        # Define custom loss function for VAE
        reconstruction_loss = losses.mse(K.flatten(x), K.flatten(x_decoded))
        reconstruction_loss *= self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

#########################################################
    def build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        for depth in self.depths:
            x = layers.Conv2D(depth, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = self.sampling([z_mean, z_log_var])
        return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

#########################################################
    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 32 * self.depths[-1], activation="relu")(latent_inputs)
        x = layers.Reshape((32, 32, self.depths[-1]))(x)
        for depth in reversed(self.depths):
            x = layers.Conv2DTranspose(depth, 3, activation="relu", strides=2, padding="same")(x)
        outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
        return models.Model(latent_inputs, outputs, name="decoder")

#########################################################
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

#########################################################


