import tensorflow as tf
import config

from model_utils import calculate_conv_output_size


n_x = config.IMAGE_PXL_SIZE_X
n_y = config.IMAGE_PXL_SIZE_Y
n_z = config.SLICES

# This handles padding in both convolution and pooling layers
strides = [[1, 1, 1],
           [2, 4, 4],
           [1, 1, 1],
           [2, 2, 2],
           [1, 1, 1],
           [1, 1, 1],
           [2, 2, 2]]

filters = [[3, 5, 5],
            [3, 5, 5],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]]
            
padding_types = ['VALID'] * 7


# Default network config used with more slices
# and larger convololution stride on first layer
additional_layers_config = {
    'weights': [
        # Convolution layers
        ('wc1', tf.truncated_normal([3, 5, 5, config.NUM_CHANNELS, 16], stddev=0.01)),
        ('wc2', tf.truncated_normal([3, 3, 3, 16, 64], stddev=0.01)),
        ('wc3', tf.truncated_normal([3, 3, 3, 64, 64], stddev=0.01)),
        ('wc4', tf.truncated_normal([3, 3, 3, 64, 32], stddev=0.01)),
        # Fully connected layers
        ('wd1', tf.truncated_normal([calculate_conv_output_size(n_x, n_y, n_z, 
                                                                strides, 
                                                                filters,
                                                                padding_types, 
                                                                32), 
                                    100], stddev=0.01)),
        ('wd2', tf.truncated_normal([100, 50], stddev=0.01)),
        ('wout', tf.truncated_normal([50, config.N_CLASSES], stddev=0.01))
    ],
    'biases': (
        # Convolution layers
        ('bc1', tf.zeros([16])),
        ('bc2', tf.constant(1.0, shape=[64])),
        ('bc3', tf.zeros([64])),
        ('bc4', tf.constant(1.0, shape=[32])),
        # Fully connected layers
        ('bd1', tf.constant(1.0, shape=[100])),
        ('bd2', tf.constant(1.0, shape=[50])),
        ('bout', tf.constant(1.0, shape=[config.N_CLASSES]))
    ),
    'pool_strides': [
        [1, 2, 4, 4, 1],
        [1, 2, 2, 2, 1],
        [],
        [1, 2, 2, 2, 1],
    ],
    'pool_windows': [
        [1, 3, 5, 5, 1],
        [1, 3, 3, 3, 1],
        [],
        [1, 3, 3, 3, 1],
    ],
    'strides': [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
}