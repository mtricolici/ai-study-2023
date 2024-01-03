import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics, losses
from helper import lm

##############################################################################
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
##############################################################################
def compute_loss2(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    reconstruction = model.decode(z)

    # Reconstruction Loss
    reconstruction_loss = tf.reduce_mean(
        losses.binary_crossentropy(x, reconstruction))

    # KL Divergence Loss
    kl_loss = -0.5 * tf.reduce_mean(
        logvar - tf.square(mean) - tf.exp(logvar) + 1)

    # total loss
    loss = reconstruction_loss + kl_loss

    return loss, kl_loss, reconstruction_loss
##############################################################################
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    reconstruction = model.decode(z)

    # Reconstruction Loss
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=x)
    reconstruction_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    # KL Divergence Loss
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_loss = tf.reduce_mean(logqz_x - logpz)

    # total loss
    loss = reconstruction_loss + kl_loss
    return loss, kl_loss, reconstruction_loss
##############################################################################
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, _, _ = compute_loss(model, x)
    gradients = tape.gradient(
        loss,
        model.encoder.trainable_variables + model.decoder.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients,
            model.encoder.trainable_variables + model.decoder.trainable_variables))
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
def create_metrics():
    return {
        'loss': metrics.Mean(),
        'kl_loss': metrics.Mean(),
        'rec_loss': metrics.Mean()
    }
##############################################################################
def collect_validation_loss(metrics, model, test_x):
    loss, kl_loss, rec_loss = compute_loss(model, test_x)
    metrics['loss'](loss)
    metrics['kl_loss'](kl_loss)
    metrics['rec_loss'](rec_loss)
##############################################################################
def get_loss(metrics):
    ls = []
    for k, m in metrics.items():
        v = tf.abs(m.result())
        if k == 'loss':
            loss = v
        ls.append(f'{k}: {v:.6f}')
    return loss, ", ".join(ls)
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
        mtx = create_metrics()
        for test_x in test_dataset:
            collect_validation_loss(mtx, model, test_x)

        # get loss and save model if it is better
        loss, loss_s = get_loss(mtx)
        best = '-'
        if best_loss > loss:
            best_loss = loss
            best = '+'
            model.save_model()

        # print stats
        lm(f'Epoch: {epoch:02d}# {best}{loss_s}')

##############################################################################




