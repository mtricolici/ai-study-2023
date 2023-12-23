import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
import numpy as np

from constants import *
import dataset as ds

class VAE:
#########################################################
    def __init__(self, latent_dim, input_shape):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.create_model()
#########################################################
    def save_model(self):
        self.encoder.save_weights('/content/model_encoder.h5')
        self.decoder.save_weights('/content/model_decoder.h5')

#########################################################
    def load_model(self):
        self.encoder.load_weights('/content/model_encoder.h5')
        self.decoder.load_weights('/content/model_decoder.h5')
#########################################################
    def generate_samples(self, num_samples):
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(random_latent_points)
#########################################################
    @tf.function(reduce_retracing=True)
    def _train_step(self, optimizer, x_batch, loss_metric):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_batch)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                losses.binary_crossentropy(x_batch, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            loss = reconstruction_loss + kl_loss

        grads = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        loss_metric(loss)
#########################################################
    def _train_epoch(self, optimizer, dl, loss_metric):
        for step in range(STEPS_PER_EPOCH):
            x_batch = next(dl)
            self._train_step(optimizer, x_batch, loss_metric)
#########################################################
    def train(self):
        dl = ds.data_loader()
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        loss_metric = metrics.Mean()

        for epoch in range(EPOCH):
            self._train_epoch(optimizer, dl, loss_metric)
            print(f"Epoch {epoch + 1}/{EPOCH}, Loss: {loss_metric.result()}")
#########################################################
    def create_model(self):
        # Encoder model
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(DEPTH, activation='relu', kernel_initializer='he_normal')(x)
        z_mean = layers.Dense(self.latent_dim, kernel_initializer='he_normal')(x)
        z_log_var = layers.Dense(self.latent_dim, kernel_initializer='he_normal')(x)
        z = self.reparameterize(z_mean, z_log_var)

        self.encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder model
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(DEPTH, activation='relu', kernel_initializer='he_normal')(latent_inputs)

        outputs = layers.Dense(tf.reduce_prod(self.input_shape), activation='sigmoid')(x)
        outputs = layers.Reshape(self.input_shape)(outputs)

        self.decoder = models.Model(latent_inputs, outputs, name='decoder')

#########################################################
    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

#########################################################

