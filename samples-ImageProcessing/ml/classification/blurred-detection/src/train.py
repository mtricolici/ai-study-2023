import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

import dataset as ds
from dataset import DataSet

#########################################################################
def train_model(model):

    ds = DataSet(batch_size=32, split_ratio=0.8)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint('/content/model.hdf5', save_best_only=True) #, save_weights_only=True)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        verbose=1,
        min_lr=1e-20
    )

    try:
      model.fit(
        ds.train_data(),
        steps_per_epoch=100,
        epochs=2,
        validation_data=ds.validation_data(),
        validation_steps=ds.validation_steps,
        callbacks=[checkpoint, early_stopping, lr_scheduler])
    except KeyboardInterrupt:
      print('Aborting...')

#########################################################################

