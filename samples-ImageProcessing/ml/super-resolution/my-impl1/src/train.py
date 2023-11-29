from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from dataset import dataset_loader, validation_dataset_loader, calc_validation_steps
from constants import *
from helper import psnr_metric

#########################################################
def train(model):
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
        patience=3,       # number of epochs with no improvement after which learning rate will be reduced
        verbose=1,
        mode='max',       # Change mode to 'max' since higher PSNR is better
        min_lr=0.00001    # lower bound on the learning rate
    )

    try:
      model.fit(
        dataset_loader(),
        epochs=EPOCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_dataset_loader(),
        validation_steps=val_steps,
        callbacks=[checkpoint, early_stopping, lr_scheduler])
    except KeyboardInterrupt:
      print('Aborting...')
#########################################################

