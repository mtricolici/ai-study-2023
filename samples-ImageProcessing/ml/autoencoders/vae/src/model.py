import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

import dataset as ds
from image import save_image

#########################################################
class VAE:
#########################################################
    def __init__(self):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        self.latent_dim = 128
        self.depths = [64, 128]
        self.latent_space = int(128 / 2 ** len(self.depths))
        self.learning_rate = 1e-4
        self.batch_size = 10
        self.epochs = 10
        self.steps_per_epoch = 100

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
#########################################################
    def save_model(self):
        self.encoder.save_weights('/content/encoder.h5')
        self.decoder.save_weights('/content/decoder.h5')
#########################################################
    def load_model(self):
        self.encoder.load_weights('/content/encoder.h5')
        self.decoder.load_weights('/content/decoder.h5')
#########################################################
    def generate_samples(self, num_samples):
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(random_latent_points)
#########################################################
    def build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        for depth in self.depths:
            x = layers.Conv2D(depth, 3, activation="relu", strides=2, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
#            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean", kernel_regularizer=regularizers.l2(1e-4))(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var", kernel_regularizer=regularizers.l2(1e-4))(x)
        z = self.sampling([z_mean, z_log_var])
        return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

#########################################################
    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.latent_space * self.latent_space * self.depths[-1], activation="relu", kernel_regularizer=regularizers.l2(1e-4))(latent_inputs)
        x = layers.Reshape((self.latent_space, self.latent_space, self.depths[-1]))(x)

        for depth in reversed(self.depths):
            x = layers.Conv2DTranspose(depth, 3, activation="relu", strides=2, padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
#            x = layers.UpSampling2D((2, 2))(x)

        outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
        return models.Model(latent_inputs, outputs, name="decoder")

#########################################################
    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

#########################################################
    @tf.function(reduce_retracing=True)
    def _train_step(self, optimizer, x_batch, mtx):
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

        # collect metrics
        mtx[0](loss)
        mtx[1](kl_loss)
        mtx[2](reconstruction_loss)
#########################################################
    def _train_epoch(self, optimizer, dl, mtx):
        for step in range(self.steps_per_epoch):
            x_batch = next(dl)
            self._train_step(optimizer, x_batch, mtx)
#########################################################
    def train(self):
        dl = ds.data_loader(self.batch_size)
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        mtx = [metrics.Mean(), metrics.Mean(), metrics.Mean()]

        for epoch in range(self.epochs):
            self._train_epoch(optimizer, dl, mtx)
            loss = f'Loss: {mtx[0].result()} kl_loss: {mtx[1].result()} reconstruction_loss: {mtx[2].result()}'
            print(f"Epoch {epoch + 1}/{self.epochs}, {loss}")
#########################################################

