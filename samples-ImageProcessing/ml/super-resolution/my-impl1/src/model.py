import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers as tf_l
from tensorflow.keras import models as tf_m
from tensorflow.keras import optimizers as tf_o
from tensorflow.keras import initializers as tf_i

from constants import *
from helper import psnr_metric

#########################################################
# Inspiration from: https://github.com/idealo/image-super-resolution/blob/master/ISR/models/rdn.py
#########################################################
class MyModel:

    def __init__(self):
      self.num_filters = NUM_FILTERS
      self.nr_of_colors = 3
      self.scale = SCALE_FACTOR
      self.rds_count = RDS_COUNT
      self.rds_conv_layers = RDS_CONV_LAYERS
      self.kernel_size = 3

#########################################################
    def _conv2d(self, inputs, size=None, ks=None, padding='same'):
        input_size = self.num_filters if size is None else size
        kernel_size = self.kernel_size if ks is None else ks

        return tf_l.Conv2D(
            input_size,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=tf_i.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        )(inputs)

#########################################################
    def create_model(self):
        input_layer = tf_l.Input(shape=(None, None, self.nr_of_colors))

        # First convolution
        x = self._conv2d(input_layer)
        conv1_layer = x
        # 2nd convultion
        x = self._conv2d(x)
        # Add residual blocks
        x = self._residual_blocks(x)
        # Global Feature Fusion
        x = self._conv2d(x, ks=1)
        x = self._conv2d(x)
        # Global Residual Learning for Dense Features
        x = tf_l.Add()([x, conv1_layer])
        # Upscaling
        x = self._upscaling_layers(x)
        # Compose SR image
        x = self._conv2d(x, size=self.nr_of_colors)
        model = tf_m.Model(inputs=input_layer, outputs=x)

        model.compile(
            optimizer=tf_o.Adam(learning_rate=LEARNING_RATE),
            loss='mean_squared_error',
            metrics=[psnr_metric])

        return model
#########################################################
    def _residual_blocks(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer

        for _ in range(self.rds_count):
            x = rdb_in

            for _ in range(self.rds_conv_layers):
                y = self._conv2d(x)
                y = tf_l.Activation('relu')(y)
                x = tf_l.concatenate([x, y], axis=3)

            # 1x1 convolution (Local Feature Fusion)
            x = self._conv2d(x, ks=1, padding='valid')

            # Local Residual Learning F_{i,LF} + F_{i-1}
            rdb_in = tf_l.Add()([x, rdb_in])
            rdb_concat.append(rdb_in)

        assert len(rdb_concat) == self.rds_count

        return tf_l.concatenate(rdb_concat, axis=3)
#########################################################
    def _upscaling_layers(self, x):
        x = self._conv2d(x, size=64, ks=5)
        x = tf_l.Activation('relu')(x)
        x = self._conv2d(x, size=32, ks=3)
        x = tf_l.Activation('relu')(x)

        # Upsampling
        x = self._conv2d(x, size=self.nr_of_colors * self.scale ** 2, ks=3)

        return tf_l.UpSampling2D(size=self.scale)(x)
#########################################################

