import time
import numpy as np
import tensorflow as tf

##############################################################################
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
##############################################################################
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
##############################################################################
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
##############################################################################
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > 0.5, 1.0, 0.0).astype('float32')
##############################################################################
def load_dataset(train_size = 10000, batch_size = 32, test_size = 1000):

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(batch_size))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(batch_size))

    return train_dataset, test_dataset
##############################################################################
def train(model, epochs=10):
    train_dataset, test_dataset = load_dataset()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        # training
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()

        # validation
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, loss: {}, elapsed: {}'
            .format(epoch, elbo, end_time - start_time))
##############################################################################




