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
        'params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
        'name': 'rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5',
    },
    'psnr-small': {
        'params': {'C': 3, 'D': 10, 'G': 64, 'G0': 64, 'x': 2},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
        'name': 'rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5',
    },
    'noise-cancel': {
        'params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
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

    self.name = name
    self.p = mp['params']
    self.weights_fname = mp['name']
    self.download_url = mp['url']

    self.patch_size = None # Input size.
    self.kernel_size = 3
    self.c_dim = 3 # Number of colors
    self.scale = self.p['x']

    self.model = self._create_model()
    self._download_weights()
#########################################################
  def get_initializer(self):
    return tf_i.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
#########################################################
  def __str__(self):
    return f"IsrRdn(name:'{self.name}', params:{self.p})"
#########################################################
  def _create_model(self):
    input_layer = tf_l.Input(shape=(self.patch_size, self.patch_size, self.c_dim), name='input')

    conv1_layer = tf_l.Conv2D(
      self.p['G0'],
      kernel_size=self.kernel_size,
      padding='same',
      kernel_initializer=self.get_initializer(),
    )(input_layer)

    x = tf_l.Conv2D(
      self.p['G0'],
      kernel_size=self.kernel_size,
      padding='same',
      kernel_initializer=self.get_initializer(),
    )(conv1_layer)

    FD = self._RDBs(x)

    # Global Feature Fusion
    # 1x1 Conv of concat RDB layers -> G0 feature maps
    GFF1 = tf_l.Conv2D(
      self.p['G0'],
      kernel_size=1,
      padding='same',
      kernel_initializer=self.get_initializer(),
    )(FD)

    GFF2 = tf_l.Conv2D(
      self.p['G0'],
      kernel_size=self.kernel_size,
      padding='same',
      kernel_initializer=self.get_initializer(),
    )(GFF1)

    # Global Residual Learning for Dense Features
    FDF = tf_l.Add()([GFF2, conv1_layer])

    # Upscaling
    FU = self._UPN(FDF)

    # Compose SR image
    SR = tf_l.Conv2D(
      self.c_dim,
      kernel_size=self.kernel_size,
      padding='same',
      kernel_initializer=self.get_initializer(),
    )(FU)

    return tf_m.Model(inputs=input_layer, outputs=SR)
#########################################################
  def _RDBs(self, input_layer):
    rdb_concat = list()
    rdb_in = input_layer
    for d in range(1, self.p['D'] + 1):
      x = rdb_in
      for c in range(1, self.p['C'] + 1):
        F_dc = tf_l.Conv2D(
          self.p['G'],
          kernel_size=self.kernel_size,
          padding='same',
          kernel_initializer=self.get_initializer(),
          name='F_%d_%d' % (d, c),
        )(x)
        F_dc = tf_l.Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
        # concatenate input and output of ConvRelu block
        x = tf_l.concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))

      # 1x1 convolution (Local Feature Fusion)
      x = tf_l.Conv2D(
        self.p['G0'], kernel_size=1, kernel_initializer=self.get_initializer(), name='LFF_%d' % (d)
      )(x)
      # Local Residual Learning F_{i,LF} + F_{i-1}
      rdb_in = tf_l.Add(name='LRL_%d' % (d))([x, rdb_in])
      rdb_concat.append(rdb_in)

    assert len(rdb_concat) == self.p['D']

    return tf_l.concatenate(rdb_concat, axis=3, name='LRLs_Concat')
#########################################################
  def _UPN(self, input_layer):
    x = tf_l.Conv2D(
      64,
      kernel_size=5,
      strides=1,
      padding='same',
      name='UPN1',
      kernel_initializer=self.get_initializer(),
    )(input_layer)
    x = tf_l.Activation('relu', name='UPN1_Relu')(x)
    x = tf_l.Conv2D(
      32, kernel_size=3, padding='same', name='UPN2', kernel_initializer=self.get_initializer()
    )(x)
    x = tf_l.Activation('relu', name='UPN2_Relu')(x)
    return self._upsampling_block(x)
#########################################################
  def _upsampling_block(self, input_layer):
    x = tf_l.Conv2D(
      self.c_dim * self.scale ** 2,
      kernel_size=3,
      padding='same',
      name='UPN3',
      kernel_initializer=self.get_initializer(),
    )(input_layer)
    return tf_l.UpSampling2D(size=self.scale, name='UPsample')(x)
#########################################################
  def _download_weights(self):
    weights_path = tf.keras.utils.get_file(fname=self.weights_fname, origin=self.download_url)
    self.model.load_weights(weights_path)
#########################################################

