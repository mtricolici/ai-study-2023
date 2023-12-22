import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class VAE:
    def __init__(self, latent_dim, input_shape):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = None
        self.decoder = None
        self.vae = None

    def create_model(self):
        # Encoder
        input_layer = keras.Input(shape=self.input_shape)
        x = layers.Flatten()(input_layer)
        x = layers.Dense(256, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.0)
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        self.encoder = keras.Model(input_layer, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        decoded = layers.Reshape(self.input_shape)(x)

        self.decoder = keras.Model(latent_inputs, decoded, name='decoder')

        # VAE model
        outputs = self.decoder(self.encoder(input_layer)[2])
        self.vae = keras.Model(input_layer, outputs, name='vae')

    def save_model(self, model_dir):
        self.encoder.save_weights(model_dir + '/encoder_weights.h5')
        self.decoder.save_weights(model_dir + '/decoder_weights.h5')

    def load_model(self, model_dir):
        self.create_model()
        self.encoder.load_weights(model_dir + '/encoder_weights.h5')
        self.decoder.load_weights(model_dir + '/decoder_weights.h5')


