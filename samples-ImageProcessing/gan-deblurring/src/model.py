import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model

from constants import *
from dataset import dataset_loader

class MyGanModel:

  #########################################################
  def __init(self):
    self.generator = None
    self.discriminator = None
    self.gan = None

  #########################################################
  def create(self):
    # Generator
    generator_input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, (3, 3), padding='same')(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU()(x)
    generator_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    self.generator = Model(generator_input, generator_output, name='generator')

    # Discriminator
    discriminator_input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(64, (3, 3), padding='valid')(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding='valid')(x)
    x = layers.LeakyReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    discriminator_output = layers.Dense(1, activation='sigmoid')(x)
    self.discriminator = Model(discriminator_input, discriminator_output, name='discriminator')

    # Compile the discriminator
    self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    # GAN
    gan_input = layers.Input(shape=(None, None, 3))
    generated_image = self.generator(gan_input)
    self.discriminator.trainable = False
    gan_output = self.discriminator(generated_image)
    self.gan = Model(gan_input, gan_output, name='gan')

    # Compile the GAN
    self.gan.compile(optimizer='adam', loss='binary_crossentropy')

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
    data_loader = dataset_loader()
    for epoch in range(EPOCH):
      print(f"Epoch {epoch+1}/{EPOCH}")
      self.train_one_epoch(data_loader)
      self.save()

  #########################################################
  def train_one_epoch(self, data_loader):
    for step in range(STEPS_PER_EPOCH):
      batch_input, batch_output = next(data_loader)

      g_loss = 0.1
      d_loss = 0.1

      if step % 100 == 0:
        print(f"Step {step}, Generator Loss: {g_loss}, Discriminator Loss: {d_loss}")
  #########################################################

