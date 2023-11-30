import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Activation, Input, Conv2D, Add, Lambda, BatchNormalization, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#########################################################
# Model imported from: https://github.com/idealo/image-super-resolution/blob/master/ISR/models/rdn.py
#########################################################
#        'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
def create_isr_model():
  initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None)

  LR_input = Input(shape=(None, None, 3), name='LR')

  F_m1 = Conv2D(
    64,
    kernel_size=3,
    padding='same',
    kernel_initializer=initializer,
    name='F_m1',
  )(LR_input)

  F_0 = Conv2D(
    64,
    kernel_size=3,
    padding='same',
    kernel_initializer=initializer,
    name='F_0',
  )(F_m1)

  FD = rds_block(F_0, initializer) # self._RDBs(F_0)

  # Global Feature Fusion
  # 1x1 Conv of concat RDB layers -> G0 feature maps
  GFF1 = Conv2D(
    64,
    kernel_size=1,
    padding='same',
    kernel_initializer=initializer,
    name='GFF_1',
  )(FD)

  GFF2 = Conv2D(
    64,
    kernel_size=3,
    padding='same',
    kernel_initializer=initializer,
    name='GFF_2',
  )(GFF1)

  # Global Residual Learning for Dense Features
  FDF = Add(name='FDF')([GFF2, F_m1])

  # Upscaling
  FU = _upn(FDF, initializer) # self._UPN(FDF)

  # Compose SR image
  SR = Conv2D(
    3,
    kernel_size=3,
    padding='same',
    kernel_initializer=initializer,
    name='SR',
  )(FU)

  return Model(inputs=LR_input, outputs=SR)
#########################################################
#        'arch_params': {'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2},
def rds_block(input_layer, initializer):
  self_D = 20
  self_C = 6
  self_G = 64
  self_G0 = 64

  rdb_concat = list()
  rdb_in = input_layer

  for d in range(1, self_D + 1):
    x = rdb_in
    for c in range(1, self_C + 1):
      F_dc = Conv2D(
          self_G,
          kernel_size=3,
          padding='same',
          kernel_initializer=initializer,
          name='F_%d_%d' % (d, c),
      )(x)
      F_dc = Activation('relu', name='F_%d_%d_Relu' % (d, c))(F_dc)
      # concatenate input and output of ConvRelu block
      # x = [input_layer,F_11(input_layer),F_12([input_layer,F_11(input_layer)]), F_13..]
      x = concatenate([x, F_dc], axis=3, name='RDB_Concat_%d_%d' % (d, c))

    # 1x1 convolution (Local Feature Fusion)
    x = Conv2D(
        self_G0, kernel_size=1, kernel_initializer=initializer, name='LFF_%d' % (d)
    )(x)
    # Local Residual Learning F_{i,LF} + F_{i-1}
    rdb_in = Add(name='LRL_%d' % (d))([x, rdb_in])
    rdb_concat.append(rdb_in)
  
  assert len(rdb_concat) == self_D
  
  return concatenate(rdb_concat, axis=3, name='LRLs_Concat')

#########################################################
def _upn(input_layer, initializer):
  x = Conv2D(
    64,
    kernel_size=5,
    strides=1,
    padding='same',
    name='UPN1',
    kernel_initializer=initializer,
  )(input_layer)
  x = Activation('relu', name='UPN1_Relu')(x)
  x = Conv2D(
    32, kernel_size=3, padding='same', name='UPN2', kernel_initializer=initializer
  )(x)
  x = Activation('relu', name='UPN2_Relu')(x)
  return _upsampling_block(x, initializer)
#########################################################
def _upsampling_block(input_layer, initializer):
  scale = 2
  c_dim = 3

  x = Conv2D(
    c_dim * scale ** 2,
    kernel_size=3,
    padding='same',
    name='UPN3',
    kernel_initializer=initializer,
  )(input_layer)
  return UpSampling2D(size=scale, name='UPsample')(x)
#########################################################

