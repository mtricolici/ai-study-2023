import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, losses
from tensorflow.keras import backend as K
from helper import lm

##############################################################################
def mse_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return 1000 * r_loss # Why????
##############################################################################
def kl_loss(mean, log_var):
    return -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
##############################################################################
def vae_loss(y_true, y_pred, mean, log_var):
    r_loss = mse_loss(y_true, y_pred)
    kl = kl_loss(mean, log_var)
    return  r_loss + kl
##############################################################################
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        mean, log_var = model.encoder(x, training=True)
        latent = model.final([mean, log_var])
        generated_images = model.decoder(latent, training=True)
        loss = vae_loss(x, generated_images, mean, log_var)

    train_vars = model.encoder.trainable_variables + model.decoder.trainable_variables

    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))

##############################################################################
@tf.function
def calc_validation_loss(model, x):
    with tf.GradientTape() as tape:
        mean, log_var = model.encoder(x, training=False)
        latent = model.final([mean, log_var])
        generated_images = model.decoder(latent, training=False)
        loss = vae_loss(x, generated_images, mean, log_var)
    return loss
##############################################################################
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > 0.5, 1.0, 0.0).astype('float32')
##############################################################################
def load_dataset(train_size = 60000, batch_size = 32, test_size = 10000):

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(batch_size))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(batch_size))

    return train_dataset, test_dataset
##############################################################################
def train(model, epochs=20):
    train_dataset, test_dataset = load_dataset()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    best_loss  = float('inf')

    for epoch in range(1, epochs + 1):
        # training
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)

        # validation
        lossMetric = metrics.Mean()
        for test_x in test_dataset:
            lossMetric(calc_validation_loss(model, test_x))

        # get loss and save model if it is better
        loss = tf.abs(lossMetric.result())
        best = '-'
        if best_loss > loss:
            best_loss = loss
            best = '+'
            model.save_model()

        # print stats
        lm(f'Epoch: {epoch:02d}# {best}loss:{loss:.5f}')

##############################################################################




