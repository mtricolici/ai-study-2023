import sys
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

from dataset import DataSet
from image import save_image
from train_helper import TrainHelper
from helper import lm, vae_loss, build_final

import variant_cnn
import variant_mlp
import variant_resnet

#########################################################
class VAE:
#########################################################
    def __init__(self, model_type = 'cnn'):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        self.latent_dim = 8
        self.learning_rate = 1e-5

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
        self.learning_rate_patience = 2

        # Stop traing if no improvements for this nr of epoches
        self.early_stop = 20

        self.train_helper = TrainHelper(self)

        self.model_type = model_type
        self.final   = build_final(self)

        # convolutional neural network (CNN)
        if model_type == 'cnn':
            self.f1 = 128
            self.f2 = 256
            self.u = 32
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
        return self.decoder(eps, training=False)
#########################################################
    @tf.function
    def _train_step(self, x):
        with tf.GradientTape() as tape:
            mean, log_var = self.encoder(x, training=True)
            latent = self.final([mean, log_var])
            generated_images = self.decoder(latent, training=True)
            loss = vae_loss(x, generated_images, mean, log_var)

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        return loss
#########################################################
    @tf.function
    def _calc_validation_loss(self, x):
        with tf.GradientTape() as tape:
            mean, log_var = self.encoder(x, training=False)
            latent = self.final([mean, log_var])
            generated_images = self.decoder(latent, training=False)
            loss = vae_loss(x, generated_images, mean, log_var)
        return loss
#########################################################
    def _train_epoch(self, epoch, training_data):
        loss = tf.keras.metrics.Mean()

        for step in range(self.steps_per_epoch):
            x_batch = next(training_data)
            loss(self._train_step(x_batch))
            self.train_helper.on_step_end(epoch+1, step, loss.result())

        # Return training loss for this epoch
        return tf.abs(loss.result())
#########################################################
    def _validation(self, validation_data):
        loss = tf.keras.metrics.Mean()

        for x in validation_data:
            loss(self._calc_validation_loss(x))
            self.train_helper.on_validation_step(loss.result())

        # Return validation loss for this epoch
        return tf.abs(loss.result())
#########################################################
    def create_optimizer(self, lr):
#        self.optimizer = optimizers.Adam(learning_rate=lr)
#        self.optimizer = optimizers.RMSprop(learning_rate=lr)
#        self.optimizer = optimizers.SGD(learning_rate=lr)
#        self.optimizer = optimizers.Adagrad(learning_rate=lr)
        self.optimizer = optimizers.Nadam(learning_rate=lr)
#        self.optimizer = optimizers.Adadelta(learning_rate=lr)
#########################################################
    def train(self):
        ds = DataSet(self.batch_size)

        # Create optimizer with initial learning rate
        self.create_optimizer(self.learning_rate)

        self.train_helper.training_start()

        for epoch in range(self.epochs):
            # train one epoch
            loss = self._train_epoch(epoch, ds.train_samples())
            # Invoke validation
            val_loss = self._validation(ds.validation_samples())

            must_stop = self.train_helper.on_epoch_end(epoch+1, loss, val_loss)
            if must_stop:
                break
#########################################################

