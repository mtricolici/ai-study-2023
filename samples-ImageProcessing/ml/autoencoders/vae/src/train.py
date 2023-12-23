import tensorflow as tf
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import numpy as np

import dataset as ds
from constants import *

#########################################################
def vae_loss(x, x_decoded_mean, z_log_var, z_mean):
#    z_log_var = tf.clip_by_value(z_log_var, -10, 10)  # Clip to a reasonable range
    reconstruction_loss = mse(x, x_decoded_mean)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

#########################################################
@tf.function(reduce_retracing=True)
def train_step(vae, optimizer, x_batch):
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
def train_epoch(vae, optimizer, dl):
    loss = 0.0
    for step in range(STEPS_PER_EPOCH):
        x_batch = next(dl)
        loss += train_step(vae, optimizer, x_batch)

    return loss / STEPS_PER_EPOCH
#########################################################
def train(vae):
    dl = ds.data_loader()
    optimizer = Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCH):
        loss = train_epoch(vae, optimizer, dl)
        print(f"Epoch {epoch + 1}/{EPOCH}, Loss: {loss:.4f}")
#########################################################

