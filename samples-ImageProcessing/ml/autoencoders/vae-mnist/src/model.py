import tensorflow as tf
from tensorflow.keras import layers, models

##############################################################################
class CVAE:
##############################################################################
    def __init__(self):
        self.input_shape = (28, 28, 1)
        self.latent_dim = 2
        self.filters = [64, 128]
        self.u = 7


        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
##############################################################################
    def build_encoder(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        for f in self.filters:
            x = layers.Conv2D(filters=f, kernel_size=3, strides=(2, 2), activation='relu')(x)

        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim * 2)(x)

        model = models.Model(inputs=inputs, outputs=x)
        return model
##############################################################################
    def build_decoder(self):
        inputs = tf.keras.Input(shape=(self.latent_dim,))

        du = self.u * self.u * self.filters[0]

        x = layers.Dense(units=du, activation='relu')(inputs)
        x = layers.Reshape(target_shape=(self.u, self.u, self.filters[0]))(x)

        for f in reversed(self.filters):
            x = layers.Conv2DTranspose(filters=f, kernel_size=3, strides=2,
                    padding='same', activation='relu')(x)

        x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(x)

        if x.shape[1:] != self.input_shape:
            raise ValueError(f"Decoder OUTPUT shape {x.shape[1:]} does not match input shape {self.input_shape}!!!")

        model = models.Model(inputs=inputs, outputs=x)
        return model
##############################################################################
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
##############################################################################
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
##############################################################################
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
##############################################################################
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits
##############################################################################
    def save_model(self):
        self.encoder.save_weights('/content/encoder.h5')
        self.decoder.save_weights('/content/decoder.h5')
##############################################################################
    def load_model(self):
        self.encoder.load_weights('/content/encoder.h5')
        self.decoder.load_weights('/content/decoder.h5')
##############################################################################










