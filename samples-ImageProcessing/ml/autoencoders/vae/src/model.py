import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

import dataset as ds
from image import save_image
from train_helper import TrainHelper
from helper import lm

import variant_cnn
import variant_mlp
import variant_resnet

#########################################################
class VAE:
#########################################################
    def __init__(self, model_type = 'resnet'):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        self.latent_dim = 256
        self.learning_rate = 1e-4

        # Higher alpha (near 1): Smoother learning, less sparsity, potential performance gains.
        # Lower alpha  (near 0): More sparsity, efficiency, risk of dying ReLU.
        self.relu_alpha = 0.6

        # L2 Regularization. strength of regularization.
        # Biger values: forces the model to learn simpler patterns: ex:  1e-3, 1e-2
        # Smaller values: forces the model to learn more paterns. ex: 1e-5, 1e-6
        self.l2r = 1e-7
        self.batch_size = 10
        self.epochs = 5000
        self.steps_per_epoch = 100

        # new_lr = lr * factor
        self.learning_rate_decrease_factor = 0.1
        self.minimum_learning_rate = 1e-18

        # number of epochs with no improvement then LR will be reduced
        self.learning_rate_patience = 3

        # Stop traing if no improvements for this nr of epoches
        self.early_stop = 20

        self.train_helper = TrainHelper(self)

        self.model_type = model_type

        # convolutional neural network (CNN)
        if model_type == 'cnn':
            self.depths = [32, 64]
            self.latent_space = int(128 / 2 ** len(self.depths))

            self.encoder = variant_cnn.build_encoder(self)
            self.decoder = variant_cnn.build_decoder(self)

        # fully connected neural network MLP (multi-layer perceptron)
        elif model_type == 'mlp':
            self.depths = [256, 512]
            self.encoder = variant_mlp.build_encoder(self)
            self.decoder = variant_mlp.build_decoder(self)

        elif model_type == 'resnet':
            self.rblocks = [64, 256, 512]
            self.resnet_d = 8
            self.encoder = variant_resnet.build_encoder(self)
            self.decoder = variant_resnet.build_decoder(self)
        else:
            raise ValueError(f"Unknown model type for VAE encoder/decoder :(")
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
        return self.decoder.predict(random_latent_points, verbose=0)
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

