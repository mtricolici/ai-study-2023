import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

import dataset as ds
from image import save_image

#########################################################
class SampleGenerationCallback(Callback):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def on_epoch_end(self, epoch, logs=None):
        # Save weights after each epoch
        self.vae.save_weights(f'/content/model-ep-{epoch+1}.h5')
        decoder = self.vae.get_layer('decoder')

        num_samples=3

        random_latent_vectors = tf.random.normal(shape=(num_samples, decoder.input_shape[-1]))
        samples = decoder.predict(random_latent_vectors)
        for i, img in enumerate(samples):
            path = f'/content/ep{epoch+1:03d}-s{i+1}.png'
            save_image(img, path)

#########################################################
class VAE:
#########################################################
    def __init__(self):
        self.input_shape = (128, 128, 3) # 128x128 RGB images
        self.latent_dim = 128
        self.depths = [64, 128]
        self.latent_space = int(128 / 2 ** len(self.depths))
        self.learning_rate = 1e-5
        self.batch_size = 10
        self.epochs = 1000
        self.steps_per_epoch = 100
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.create_model()
#########################################################
    def load_model(self):
        self.vae.load_weights('/content/model.h5')
        self.encoder.set_weights(self.vae.get_layer('encoder').get_weights())
        self.decoder.set_weights(self.vae.get_layer('decoder').get_weights())
#########################################################
    def generate_samples(self, num_samples):
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(random_latent_points)
#########################################################
    def train(self):
        self.vae.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        callbacks = [
            # Stop if no progress for 5 epoches
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),

            # Save best models only
            ModelCheckpoint('/content/model.h5', monitor='loss', save_best_only=True, save_weights_only=True),

            # Reduce learning rate if no progress during 1 epoches
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-30),

            # Generate some samples per epoch to see how it progress during training
            SampleGenerationCallback(self.vae),
        ]

        self.vae.fit(
          ds.data_loader(self.batch_size),
          steps_per_epoch=self.steps_per_epoch,
          epochs=self.epochs,
          verbose=1,
          callbacks=callbacks)

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


