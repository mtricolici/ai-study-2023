import tensorflow as tf

#############################################################################
def create_model():
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, None, 3)),

            tf.keras.layers.Conv2D(96, (7, 7), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(256, (5, 5), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(1024),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(2),
            tf.keras.layers.Activation('softmax'),
        ])

    model.summary()
    return model
#############################################################################
