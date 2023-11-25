from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from dataset import dataset_loader, validation_dataset_loader
from constants import *

#########################################################
def train(model):

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1)

    try:
      model.fit(
        dataset_loader(),
        epochs=EPOCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_dataset_loader(),
        validation_steps=VALIDATION_STEPS,
        callbacks=[checkpoint, early_stopping])

      #model.save(MODEL_SAVE_PATH)
    except KeyboardInterrupt:
      print('Aborting...')
#########################################################

