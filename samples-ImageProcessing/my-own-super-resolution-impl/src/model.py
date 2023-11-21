import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model

from constants import *

#########################################################
def edsr_model(num_res_blocks=16, num_filters=64):
    if DYNAMIC_MODEL:
        # Any image size is accepted as input!
        input_layer = Input(shape=(None, None, 3), name='input_layer')
    else:
        # Exact size is accepted as input
        input_layer = Input(shape=(INPUT_SIZE[1], INPUT_SIZE[0], 3), name='input_layer')

    # First convolution
    x = Conv2D(num_filters, 3, padding='same', name='conv_initial')(input_layer)

    # Residual blocks
    for i in range(num_res_blocks):
        x = residual_block(x, num_filters, name=f'res_block_{i+1}')

    # Output convolution
    x = Conv2D(3 * (SCALE_FACTOR ** 2), 3, padding='same', name='conv_output')(x)
    x = Lambda(lambda x: tf.nn.depth_to_space(x, SCALE_FACTOR), name='pixel_shuffle')(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=x)
    return model

#########################################################
def residual_block(input_tensor, num_filters, name):
    """
    Create a residual block for EDSR.

    Args:
    input_tensor (Tensor): Input tensor for the residual block.
    num_filters (int): Number of filters in the block.

    Returns:
    Tensor: Output tensor of the residual block.
    """
    x = Conv2D(num_filters, 3, padding='same', activation='relu', name=f'{name}_conv1')(input_tensor)
    x = Conv2D(num_filters, 3, padding='same', name=f'{name}_conv2')(x)
    x = Add(name=f'{name}_add')([input_tensor, x])
    return x

#########################################################

