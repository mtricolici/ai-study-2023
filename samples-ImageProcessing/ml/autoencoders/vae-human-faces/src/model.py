import sys
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

from dataset import DataSet
from image import save_image
from train_helper import TrainHelper
from helper import lm

import variant_cnn
import variant_mlp
import variant_resnet

#########################################################
class VAE:
#########################################################
    def __init__(self, model_type = 'cnn'):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        self.latent_dim = 64
        self.learning_rate = 1e-4

        # Higher alpha (near 1): Smoother learning, less sparsity, potential performance gains.
        # Lower alpha  (near 0): More sparsity, efficiency, risk of dying ReLU.
        self.relu_alpha = 0.3

        # L2 Regularization. strength of regularization.
        # Biger values: forces the model to learn simpler patterns: ex:  1e-3, 1e-2
        # Smaller values: forces the model to learn more paterns. ex: 1e-5, 1e-6
        self.l2r = 1e-5
        self.batch_size = 10
        self.epochs = 5000
        self.steps_per_epoch = 100

        # new_lr = lr * factor
        self.learning_rate_decrease_factor = 0.5
        self.minimum_learning_rate = 1e-18

        # number of epochs with no improvement then LR will be reduced
        self.learning_rate_patience = 3

        # Stop traing if no improvements for this nr of epoches
        self.early_stop = 20

        self.train_helper = TrainHelper(self)

        self.model_type = model_type

        # convolutional neural network (CNN)
        if model_type == 'cnn':
            self.depths = [64, 128]
            self.latent_space = int(128 / 2 ** len(self.depths))

            self.encoder = variant_cnn.build_encoder(self)
            self.decoder = variant_cnn.build_decoder(self)

        # fully connected neural network MLP (multi-layer perceptron)
        elif model_type == 'mlp':
            self.depths = [256, 512]
            self.encoder = variant_mlp.build_encoder(self)
            self.decoder = variant_mlp.build_decoder(self)

        elif model_type == 'resnet':
            self.rblocks = [64, 128]
            self.resnet_d = 4
            self.resnet_dense = 512
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
        eps = np.random.normal(size=(num_samples, self.latent_dim))
        return self.sample(eps)
#########################################################
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            print('vae.sample. eps is NONE!')
            sys.exit(1)
        return self.decode(eps, apply_sigmoid=True)
#########################################################
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
#########################################################
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
#########################################################
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits
#########################################################
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
#########################################################
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)

        # Reconstruction Loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=x)
        reconstruction_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

        # KL Divergence Loss
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        kl_loss = tf.reduce_mean(logqz_x - logpz)

        # total loss
        loss = reconstruction_loss + kl_loss
        return loss, kl_loss, reconstruction_loss
#########################################################
    @tf.function
    def _train_step(self, optimizer, x):
        with tf.GradientTape() as tape:
            loss, kl_loss, reconstruction_loss = self.compute_loss(x)

        grads = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return loss, kl_loss, reconstruction_loss

#########################################################
    def _train_epoch(self, epoch, optimizer, training_data):
        for step in range(self.steps_per_epoch):
            x_batch = next(training_data)
            loss, kl, rl = self._train_step(optimizer, x_batch)

            self.train_helper.on_step_end(epoch+1, step, loss, kl, rl)
#########################################################
    def train(self):
        ds = DataSet(self.batch_size)
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        self.train_helper.training_start(optimizer)

        for epoch in range(self.epochs):
            # train 1 epoch
            self._train_epoch(epoch, optimizer, ds.train_samples())
            # Invoke validation
#            for val_batch in ds.validation_samples():
            must_stop = self.train_helper.on_epoch_end(epoch+1)
            if must_stop:
                break
#########################################################

