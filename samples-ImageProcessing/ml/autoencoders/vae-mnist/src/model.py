import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

##############################################################################
class CVAE:
##############################################################################
    def __init__(self):
        self.input_shape = (28, 28, 1)
        self.latent_dim = 2
        self.f1 = 32
        self.f2 = 64
        self.u  = 7
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.final   = self.build_final()
##############################################################################
    def build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        x = layers.Conv2D(self.f1, 3, 1, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        for _ in range(2):
            x = layers.Conv2D(self.f2, 3, 2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

        x = layers.Conv2D(self.f2, 3, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)

        mean = layers.Dense(self.latent_dim, name='mean')(x)
        log_var = layers.Dense(self.latent_dim, name='log_var')(x)

        return models.Model(inputs, (mean, log_var), name="encoder")
##############################################################################
    def build_decoder(self):
        inputs = tf.keras.Input(shape=(self.latent_dim,))

        x = layers.Dense(self.u * self.u * self.f2)(inputs)
        x = layers.Reshape((self.u, self.u, self.f2))(x)

        x = layers.Conv2DTranspose(self.f2, 3, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(self.f2, 3, 2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(self.f1, 3, 2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(1, 3, 1, padding='same', activation='sigmoid')(x)

        if x.shape[1:] != self.input_shape:
            raise ValueError(f"Decoder OUTPUT shape {x.shape[1:]} does not match input shape {self.input_shape}!!!")

        return models.Model(inputs, x, name="decoder")
##############################################################################
    def sampling_model(self, distribution_params):
        mean, log_var = distribution_params
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
        return mean + K.exp(log_var / 2) * epsilon
##############################################################################
    def build_final(self):
        mean    = tf.keras.Input(shape=(self.latent_dim,))
        log_var = tf.keras.Input(shape=(self.latent_dim,))

        out = layers.Lambda(self.sampling_model)([mean, log_var])
        return tf.keras.Model([mean,log_var], out)
##############################################################################
    @tf.function
    def sample(self, latent):
        if latent is None:
            print('VAE.sample latent is NONE!')
            sys.exit(1)
        return self.decoder(latent, training=False)
##############################################################################
    def save_model(self):
        self.encoder.save_weights('/content/encoder.h5')
        self.decoder.save_weights('/content/decoder.h5')
##############################################################################
    def load_model(self):
        self.encoder.load_weights('/content/encoder.h5')
        self.decoder.load_weights('/content/decoder.h5')
##############################################################################










