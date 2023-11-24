from tensorflow.keras.callbacks import ModelCheckpoint

from dataset import dataset_loader, validation_dataset_loader
from constants import *

#########################################################
def train(model):
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    try:
      model.fit(
        dataset_loader(),
        epochs=EPOCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_dataset_loader(),
        validation_steps=VALIDATION_STEPS,
        callbacks=[checkpoint])

      #model.save(MODEL_SAVE_PATH)
    except KeyboardInterrupt:
      print('Aborting...')
#########################################################

