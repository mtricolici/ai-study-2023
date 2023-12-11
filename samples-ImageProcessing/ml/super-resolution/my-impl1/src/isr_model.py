import tensorflow as tf
from tensorflow.keras import layers as tf_l
from tensorflow.keras import models as tf_m
from tensorflow.keras import optimizers as tf_o
from tensorflow.keras import initializers as tf_i

#########################################################
# Model imported from: https://github.com/idealo/image-super-resolution/blob/master/ISR/models/rdn.py
#########################################################
ISR_PARAMS = {
    'psnr-large': {
        'params': {'rds_conv_layers': 6, 'rds_count': 20, 'scale': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
        'name': 'rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
    },
    'psnr-small': {
        'params': {'rds_conv_layers': 3, 'rds_count': 10, 'scale': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
        'name': 'rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
    },
    'noise-cancel': {
        'params': {'rds_conv_layers': 6, 'rds_count': 20, 'scale': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
        'name': 'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5',
    }
}
#########################################################
class IsrRdn:
#########################################################
  def __init__(self, name):
    mp = ISR_PARAMS.get(name, None)
    if mp is None:
      raise Exception(f'IsrRDN unknown name "{name}"!')

    self.num_filters = 64
    self.nr_of_colors = 3

    self.name = name
    self.params = mp['params']
    self.weights_fname = mp['name']
    self.download_url = mp['url']

    self.scale           = self.params['scale']
    self.rds_count       = self.params['rds_count'] #  number of Residual Dense Blocks (RDB) insider each RRDB
    self.rds_conv_layers = self.params['rds_conv_layers'] # number of convolutional layers stacked inside a RDB
    self.kernel_size = 3

    self.model = self._create_model()
    self._download_weights()
#########################################################
  def get_initializer(self):
    return tf_i.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
#########################################################
  def __str__(self):
    return f"IsrRdn(name:'{self.name}', params:{self.params})"
#########################################################
  def _create_model(self):
    input_layer = tf_l.Input(shape=(None, None, self.nr_of_colors))

    # First convolution
    x = tf_l.Conv2D(self.num_filters, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.get_initializer())(input_layer)
    conv1_layer = x

    # 2nd convultion
    x = tf_l.Conv2D(self.num_filters, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.get_initializer())(x)

    # Add residual blocks
    x = self._residual_blocks(x)

    # Global Feature Fusion
    x = tf_l.Conv2D(self.num_filters, kernel_size=1, padding='same', kernel_initializer=self.get_initializer())(x)
    x = tf_l.Conv2D(self.num_filters, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.get_initializer())(x)

    # Global Residual Learning for Dense Features
    x = tf_l.Add()([x, conv1_layer])

    # Upscaling
    x = self._upscaling_layers(x)

    # Compose SR image
    x = tf_l.Conv2D(self.nr_of_colors, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.get_initializer())(x)

    return tf_m.Model(inputs=input_layer, outputs=x)
#########################################################
  def _residual_blocks(self, input_layer):
    rdb_concat = list()
    rdb_in = input_layer

    for _ in range(self.rds_count):
      x = rdb_in

      for _ in range(self.rds_conv_layers):
        F_dc = tf_l.Conv2D(self.num_filters, kernel_size=self.kernel_size, padding='same', kernel_initializer=self.get_initializer())(x)
        F_dc = tf_l.Activation('relu')(F_dc)
        x = tf_l.concatenate([x, F_dc], axis=3)

      # 1x1 convolution (Local Feature Fusion)
      x = tf_l.Conv2D(self.num_filters, kernel_size=1, kernel_initializer=self.get_initializer())(x)

      # Local Residual Learning F_{i,LF} + F_{i-1}
      rdb_in = tf_l.Add()([x, rdb_in])
      rdb_concat.append(rdb_in)

    assert len(rdb_concat) == self.rds_count

    return tf_l.concatenate(rdb_concat, axis=3)
#########################################################
  def _upscaling_layers(self, input_layer):
    x = tf_l.Conv2D(64, kernel_size=5, strides=1, padding='same', kernel_initializer=self.get_initializer())(input_layer)
    x = tf_l.Activation('relu')(x)
    x = tf_l.Conv2D(32, kernel_size=3, padding='same', kernel_initializer=self.get_initializer())(x)
    x = tf_l.Activation('relu')(x)

    # Upsampling
    in_size = self.nr_of_colors * self.scale ** 2
    x = tf_l.Conv2D( in_size, kernel_size=3, padding='same', kernel_initializer=self.get_initializer())(x)

    return tf_l.UpSampling2D(size=self.scale)(x)
#########################################################
  def _download_weights(self):
    weights_path = tf.keras.utils.get_file(fname=self.weights_fname, origin=self.download_url)
    self.model.load_weights(weights_path)
#########################################################

