import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from constants import *
from dataset import train_data_loader, validation_data_loader, calc_validation_steps
from image import load_image, save_image

#########################################################
def model_create():
  model = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(*INPUT_SIZE[::-1], 3)),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    Conv2DTranspose(3, (3, 3), padding='same')
  ])
  model.compile(optimizer=Adam(LEARNING_RATE), loss='mean_squared_error')
  return model

#########################################################
def train_model(model):
  val_steps = calc_validation_steps()

  checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
  early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1)

  try:
    model.fit(
      train_data_loader(),
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=EPOCH,
      validation_data=validation_data_loader(),
      validation_steps=val_steps,
      callbacks=[checkpoint, early_stopping])
  except KeyboardInterrupt:
    print('Aborting...')

#########################################################
def unblure_image(model, input_path, output_path):
  img = load_image(input_path)
  img = tf.expand_dims(img, axis=0) # Add batch dimension
  
  out = model.predict(img)
  out = tf.squeeze(out, axis=0) # Remove batch dimension
  save_image(out, output_path)

#########################################################

