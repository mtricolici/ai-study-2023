import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from constants import *
from dataset import train_data_loader, validation_data_loader, calc_validation_steps
from image import load_image, save_image

#########################################################
def train_model(model):
  val_steps = calc_validation_steps()

  checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)

  early_stopping = EarlyStopping(
    monitor='val_psnr_metric',
    patience=EARLY_STOPPING_PATIENCE,
    verbose=1,
    mode='max'  # Change mode to 'max' since higher PSNR is better!
  )

  lr_scheduler = ReduceLROnPlateau(
      monitor='val_psnr_metric',  # Monitor PSNR instead of loss
      factor=0.1,       # new_lr = lr * factor
      patience=8,       # number of epochs with no improvement after which learning rate will be reduced
      verbose=1,
      mode='max',       # Change mode to 'max' since higher PSNR is better
      min_lr=0.00001    # lower bound on the learning rate
  )

  try:
    model.fit(
      train_data_loader(),
      steps_per_epoch=STEPS_PER_EPOCH,
      epochs=EPOCH,
      validation_data=validation_data_loader(),
      validation_steps=val_steps,
      callbacks=[checkpoint, early_stopping, lr_scheduler])
  except KeyboardInterrupt:
    print('Aborting...')

#########################################################
def colorize_image(model, input_path, output_path):
  img = load_image(input_path, 1)
  img = tf.expand_dims(img, axis=0) # Add batch dimension

  out = model.predict(img)
  out = tf.squeeze(out, axis=0) # Remove batch dimension
  save_image(out, output_path)

#########################################################

