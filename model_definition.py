import tensorflow as tf

import config
from model_utils import calculate_conv_output_size

# Parameters
depth = 16
second_depth = 64
third_depth = 64
last_depth = 32

hidden = 100
second_hidden = 50

# Convolution filter size on first layer
first_kernel_size_x = 5
first_kernel_size_y = 5
first_kernel_size_z = 4

# Max pooling window size and stride on first layer
first_pool_stride_x = 3
first_pool_window_x = 4
first_pool_stride_y = 3
first_pool_window_y = 4
first_pool_stride_z = 2
first_pool_window_z = 3

# Convolution filter size for the rest of the layers
kernel_size = 3

# Max pooling window size and stride for the other layers
pool_window = 3
pool_stride = 2

# Network Parameters
n_x = config.IMAGE_PXL_SIZE_X
n_y = config.IMAGE_PXL_SIZE_Y
n_z = config.SLICES
num_channels = 1
n_input = n_x * n_y * n_z
n_classes = 2
dropout = 0.5 # Dropout, probability to keep units


# This handles padding in both convolution and pooling layers
strides = [[2, 2, 2],
           [first_pool_stride_z, first_pool_stride_x, first_pool_stride_y],
           [1, 1, 1],
           [pool_stride, pool_stride, pool_stride],
           [1, 1, 1],
           [1, 1, 1],
           [pool_stride, pool_stride, pool_stride]]

filters = [[first_kernel_size_z, first_kernel_size_x, first_kernel_size_y],
            [first_pool_window_z, first_pool_window_x, first_pool_window_y],
            [kernel_size, kernel_size, kernel_size],
            [pool_window, pool_window, pool_window],
            [kernel_size, kernel_size, kernel_size],
            [kernel_size, kernel_size, kernel_size],
            [pool_window, pool_window, pool_window]]
padding_types = ['VALID'] * 7

new_x, new_y, new_z = calculate_conv_output_size(n_x, n_y, n_z, 
                                                 strides, 
                                                 filters,
                                                 padding_types)

out_conv_size = int(new_x * new_y * new_z * last_depth)

# Default network config used with more slices
# and larger convololution stride on first layer
default_config = {
    'weights': [
        # Convolution layers
        tf.Variable(tf.truncated_normal([first_kernel_size_z, first_kernel_size_x, first_kernel_size_y, num_channels, depth], 
        stddev=0.01), name='wc1'),
        tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size, depth, second_depth], 
        stddev=0.01), name='wc2'),
        tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size, second_depth, third_depth], 
        stddev=0.01), name='wc3'),
        tf.Variable(tf.truncated_normal([kernel_size, kernel_size, kernel_size, third_depth, last_depth], 
        stddev=0.01), name='wc4'),
        # Fully connected layers
        tf.Variable(tf.truncated_normal([out_conv_size, hidden], stddev=0.01), name='wd1'),
        tf.Variable(tf.truncated_normal([hidden, second_hidden], stddev=0.01), name='wd2'),
        tf.Variable(tf.truncated_normal([second_hidden, n_classes], stddev=0.01), name='wout')
    ],
    'biases': [
        # Convolution layers
        tf.Variable(tf.zeros([depth]), name='bc1'),
        tf.Variable(tf.constant(1.0, shape=[second_depth]), name='bc2'),
        tf.Variable(tf.zeros([third_depth]), name='bc3'),
        tf.Variable(tf.constant(1.0, shape=[last_depth]), name='bc4'),
        # Fully connected layers
        tf.Variable(tf.constant(1.0, shape=[hidden]), name='bd1'),
        tf.Variable(tf.constant(1.0, shape=[second_hidden]), name='bd2'),
        tf.Variable(tf.constant(1.0, shape=[n_classes]), name='bout')
    ],
    'pool_strides': [
        [1, first_pool_stride_z, first_pool_stride_x, first_pool_stride_y, 1],
        [1, pool_stride, pool_stride, pool_stride, 1],
        [],
        [1, pool_stride, pool_stride, pool_stride, 1],
    ],
    'pool_windows': [
        [1, first_pool_window_z, first_pool_window_x, first_pool_window_y, 1],
        [1, pool_window, pool_window, pool_window, 1],
        [],
        [1, pool_window, pool_window, pool_window, 1],
    ],
    'strides': [
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
}