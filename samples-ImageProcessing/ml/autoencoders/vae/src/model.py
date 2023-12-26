import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

import dataset as ds
from image import save_image
from train_helper import TrainHelper
from helper import lm

#########################################################
class VAE:
#########################################################
    def __init__(self):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        #
        #      latent-dim: 256, depths=[ 32, 64], batchs:10 => 1377MiB VRAM
        #      latent-dim: 128, dephts=[ 64,128], batchs:10 => 2333MiB VRAM
        #      latent-dim: 128, dephts=[128,256], batchs:10 => 3560MiB VRAM
        #
        self.latent_dim = 256
        self.depths = [32, 64]
        self.latent_space = int(128 / 2 ** len(self.depths))
        self.learning_rate = 3e-4

        # Higher alpha (near 1): Smoother learning, less sparsity, potential performance gains.
        # Lower alpha  (near 0): More sparsity, efficiency, risk of dying ReLU.
        self.relu_alpha = 0.7

        # L2 Regularization. strength of regularization.
        # Biger values: forces the model to learn simpler patterns: ex:  1e-3, 1e-2
        # Smaller values: forces the model to learn more paterns. ex: 1e-5, 1e-6
        self.l2r = 1e-6
        self.batch_size = 32
        self.epochs = 5000
        self.steps_per_epoch = 100

        # new_lr = lr * factor
        self.learning_rate_decrease_factor = 0.1
        self.minimum_learning_rate = 1e-18

        # number of epochs with no improvement then LR will be reduced
        self.learning_rate_patience = 3

        # Stop traing if no improvements for this nr of epoches
        self.early_stop = 10

        self.train_helper = TrainHelper(self)

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
            x = layers.Conv2D(depth, 3, activation=layers.LeakyReLU(alpha=self.relu_alpha), strides=2, padding="same", kernel_regularizer=regularizers.l2(self.l2r))(x)
            x = layers.BatchNormalization()(x)
#            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Flatten()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean", kernel_regularizer=regularizers.l2(self.l2r))(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var", kernel_regularizer=regularizers.l2(self.l2r))(x)
        z = self.sampling([z_mean, z_log_var])
        return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

#########################################################
    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.latent_space * self.latent_space * self.depths[-1], activation=layers.LeakyReLU(alpha=self.relu_alpha),
                         kernel_regularizer=regularizers.l2(self.l2r))(latent_inputs)
        x = layers.Reshape((self.latent_space, self.latent_space, self.depths[-1]))(x)

        for depth in reversed(self.depths):
            x = layers.Conv2DTranspose(depth, 3, activation=layers.LeakyReLU(alpha=self.relu_alpha),
                    strides=2, padding="same", kernel_regularizer=regularizers.l2(self.l2r))(x)
            x = layers.BatchNormalization()(x)
#            x = layers.UpSampling2D((2, 2))(x)

        outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(self.l2r))(x)

        if outputs.shape[1:] != self.input_shape:
            raise ValueError(f"Decoder OUTPUT shape {outputs.shape[1:]} does not match input shape {self.input_shape}!!!")

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
    def _train_step(self, optimizer, x_batch):
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

        return loss, kl_loss, reconstruction_loss

#########################################################
    def _train_epoch(self, epoch, optimizer, dl):
        for step in range(self.steps_per_epoch):
            x_batch = next(dl)
            loss, kl, rl = self._train_step(optimizer, x_batch)

            self.train_helper.on_step_end(epoch+1, step, loss, kl, rl)
#########################################################
    def train(self):
        dl = ds.data_loader(self.batch_size)
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        self.train_helper.training_start(optimizer)

        for epoch in range(self.epochs):
            self._train_epoch(epoch, optimizer, dl)
            must_stop = self.train_helper.on_epoch_end(epoch+1)
            if must_stop:
                break
#########################################################

