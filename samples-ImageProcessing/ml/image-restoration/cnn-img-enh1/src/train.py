import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from constants import *
from dataset import train_data_loader, validation_data_loader, calc_validation_steps
from image import load_image, save_image

import models as m

#########################################################
def model_create(t="resnet"):
  if t == "simple":
    return m.simple.model_create()
  elif t == "resnet":
    return m.resnet.model_create()
  elif t == "dncnn":
    return m.dncnn.model_create()

  print(f"Error: unknown model {t}")

#########################################################
def train_model(model):
  val_steps = calc_validation_steps()

  checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, save_weights_only=True)

  early_stopping = EarlyStopping(
    monitor='val_psnr_metric',
    patience=EARLY_STOPPING_PATIENCE,
    verbose=1,
    mode='max'
  )

  lr_scheduler = ReduceLROnPlateau(
      monitor='val_psnr_metric',  # Monitor PSNR instead of loss
      factor=0.1,       # new_lr = lr * factor
      patience=5,       # number of epochs with no improvement after which learning rate will be reduced
      verbose=1,
      mode='max',       # Change mode to 'max' since higher PSNR is better
      min_lr=1e-11    # lower bound on the learning rate
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
def restore_image(model, input_path, output_path, iterations):
  print(f'Restoring image {input_path} into {output_path}. iterations: {iterations} ...')
  img = load_image(input_path)
  img = tf.expand_dims(img, axis=0) # Add batch dimension

  out = img

  for _ in range(iterations):
    out = model.predict(out)

  out = tf.squeeze(out, axis=0) # Remove batch dimension
  save_image(out, output_path)

#########################################################
def demo_single(model, input_path, output_path, iterations):
  original_img = load_image(input_path)

  img = tf.expand_dims(original_img, axis=0) # Add batch dimension
  out = img

  for _ in range(iterations):
    out = model.predict(out)

  out = tf.squeeze(out, axis=0) # Remove batch dimension

  out = np.hstack((original_img, out.numpy()))
  save_image(out, output_path)
#########################################################
def restore_many(model, iterations):
  in_files  = [os.path.join('/output/inputs', f) for f in os.listdir('/output/inputs') if f.endswith(".png")]
  out_files = [f.replace("/inputs/", "/outputs/") for f in in_files]

  for in_file, out_file in zip(in_files, out_files):
    print(f'processing {in_file} > {out_file} (iterations: {iterations})')
    demo_single(model, in_file, out_file, iterations)

#########################################################


