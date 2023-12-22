import tensorflow as tf
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import numpy as np

import dataset as ds
from constants import *

#########################################################
def vae_loss(x, x_decoded_mean, z_log_var, z_mean):
    reconstruction_loss = mse(x, x_decoded_mean)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

#########################################################
def train(vae):
    dl = ds.data_loader()
    optimizer = Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCH):
        print(f"Epoch {epoch + 1}/{EPOCH}")

        for step in range(STEPS_PER_EPOCH):
            x_batch = next(dl)
            with tf.GradientTape() as tape:
                x_decoded = vae.vae(x_batch)
                loss = vae_loss(x_batch, x_decoded, *vae.encoder(x_batch)[1:])

            gradients = tape.gradient(loss, vae.vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.vae.trainable_variables))

        avgl = np.mean(loss.numpy())
        print(f"Epoch {epoch + 1}/{EPOCH}, Loss: {avgl:.4f}")
        tf.keras.backend.clear_session()
#########################################################

