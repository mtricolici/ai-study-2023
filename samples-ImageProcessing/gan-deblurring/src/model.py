import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from constants import *
from dataset import dataset_loader
from image import load_image, save_image

class MyGanModel:

  #########################################################
  def __init__(self):
    self.generator = None
    self.discriminator = None
    self.gan = None
    self.hidden = 3
    self.sn = 32 # start neurons

  #########################################################
  def create_generator(self):
    i = layers.Input(shape=(None, None, 3))
    x = i
    n = self.sn
    for l in range(self.hidden):
      x = layers.Conv2D(n, (3, 3), padding='same')(x)
      x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU()(x)
      n = n*2

    o = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    self.generator = Model(i, o, name='generator')

  #########################################################
  def create_discriminator(self):
    i = layers.Input(shape=(None, None, 3))
    x = i
    n = self.sn
    for l in range(self.hidden):
      x = layers.Conv2D(n, (3, 3), padding='valid')(x)
      if l > 0:
        x = layers.BatchNormalization()(x)
      x = layers.LeakyReLU()(x)
      n = n*2
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(1, activation='sigmoid')(x)
    self.discriminator = Model(i, o, name='discriminator')
    self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE))
    self.discriminator.trainable = False

  #########################################################
  def create_gan(self):
    i = layers.Input(shape=(None, None, 3))
    generated_image = self.generator(i)
    o = self.discriminator(generated_image)
    self.gan = Model(i, o, name='gan')
    self.gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE))

  #########################################################
  def create(self):
    self.create_generator()
    self.create_discriminator()
    self.create_gan()

  #########################################################
  def check_initialized(self):
    if self.generator is None or self.discriminator is None or self.gan is None:
      print("Erorr: model is NULL!")
      os.exit(1)

  #########################################################
  def save(self):
    self.check_initialized()
    self.generator.save(f"{MODEL_SAVE_PATH}/model-generator.keras")
    self.discriminator.save(f"{MODEL_SAVE_PATH}/model-discriminator.keras")
    self.gan.save(f"{MODEL_SAVE_PATH}/model-gan.keras")
  #########################################################
  def load(self):
    self.generator = load_model(f"{MODEL_SAVE_PATH}/model-generator.keras")
    self.discriminator = load_model(f"{MODEL_SAVE_PATH}/model-discriminator.keras")
    self.gan = load_model(f"{MODEL_SAVE_PATH}/model-gan.keras")
  #########################################################
  def train(self):
    for epoch in range(EPOCH):
      print(f"Epoch {epoch+1}/{EPOCH}")
      self.train_one_epoch()
      self.save()

  #########################################################
  def train_one_epoch(self):
    for step in range(STEPS_PER_EPOCH):
      blurred_images, sharp_images = dataset_loader()

      # Generate deblurred images
      deblurred_images = self.generator.predict(blurred_images, verbose=0)

      # Labels for real and fake images
      real_labels = tf.ones((sharp_images.shape[0], 1))
      fake_labels = tf.zeros((deblurred_images.shape[0], 1))

      #### Train the discriminator
      self.discriminator.trainable = True

      d_loss_real = self.discriminator.train_on_batch(sharp_images, real_labels)
      d_loss_fake = self.discriminator.train_on_batch(deblurred_images, fake_labels)
      d_loss = 0.5 * tf.math.add(d_loss_real, d_loss_fake)
      self.discriminator.trainable = False

      # Train the generator
      # The generator tries to make the discriminator label the deblurred images as real
      g_loss = self.gan.train_on_batch(blurred_images, real_labels)

      if step % 100 == 0 or step == STEPS_PER_EPOCH - 1:
        print(f"Step {step}, Generator Loss: {g_loss}, Discriminator Loss: {d_loss}")
  #########################################################
  def process_image(self, input_path, output_path):
    self.check_initialized()
    img = load_image(input_path)
    img = tf.expand_dims(img, axis=0) # Add batch dimension

    out = self.generator.predict(img)

    out = tf.squeeze(out, axis=0)  # Remove batch dimension
    save_image(out, output_path)

