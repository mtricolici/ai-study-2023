from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from dataset import dataset_loader
from constants import *

#########################################################
def train(model):
    model.compile(optimizer=Adam(LEARNING_RATE), loss='mean_squared_error')
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=SAVE_BEST_ONLY)
    try:
      model.fit(
        dataset_loader(),
        epochs=EPOCH,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[checkpoint])
    except KeyboardInterrupt:
      model.save(MODEL_SAVE_PATH)
      print('Model saved to: "{}./*"'.format(MODEL_SAVE_PATH))

#########################################################

