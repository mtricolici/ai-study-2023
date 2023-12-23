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
@tf.function(reduce_retracing=True)
def train_epoch(vae, optimizer, dl):
    for step in range(STEPS_PER_EPOCH):
        x_batch = next(dl)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = vae.encoder(x_batch)
            reconstruction = vae.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x_batch, reconstruction)
            )
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(loss, vae.encoder.trainable_variables + vae.decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.encoder.trainable_variables + vae.decoder.trainable_variables))

    return loss
#########################################################
def train(vae):
    dl = ds.data_loader()
    optimizer = Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCH):
        loss = train_epoch(vae, optimizer, dl)
        tf.print(f"Epoch {epoch + 1}/{EPOCH}, Loss: {loss:.4f}")
#########################################################

