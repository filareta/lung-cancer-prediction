import tensorflow as tf

import config
from model_utils import calculate_conv_output_size

# Parameters
learning_rate = 0.001
batch_size = 1

depth = 16
second_depth = 64
third_depth = 32
last_depth = 32

hidden = 100
second_hidden = 50

# Convolution filter size on first layer
first_kernel_size_x = 5
first_kernel_size_y = 5
first_kernel_size_z = 3

# Max pooling window size and stride on first layer
first_pool_stride_x = 4
first_pool_window_x = 5
first_pool_stride_y = 4
first_pool_window_y = 5
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
dropout = 0.7 # Dropout, probability to keep units

image_tensor_shape = [-1, n_z, n_x, n_y, num_channels]

# tf Graph input
x = tf.placeholder(tf.float32, shape=(batch_size, n_z, n_x, n_y, num_channels), 
    name='train_input')
y = tf.placeholder(tf.int32, shape=(batch_size,), name='label')
keep_prob = tf.placeholder(tf.float32, name='dropout') #dropout (keep probability)

tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), 
    name='validation_set')
tf_test_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), 
    name='test_set')


# This handles padding in both convolution and pooling layers
strides = [[1, 1, 1],
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
padding_types = ['VALID', 'VALID', 'VALID', 
                'VALID', 'VALID', 'VALID', 'VALID']

new_x, new_y, new_z = calculate_conv_output_size(n_x, n_y, n_z, 
                                                 strides, 
                                                 filters,
                                                 padding_types)

out_conv_size = int(new_x * new_y * new_z * last_depth)
print("Last conv net output size should be: ", out_conv_size)


# Store layers weight & bias
weights = {
    # Convolution layers
    'wc1': tf.Variable(tf.random_normal([first_kernel_size_z, first_kernel_size_x, first_kernel_size_y, num_channels, depth], 
        stddev=0.01), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, depth, second_depth], 
        stddev=0.01), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, second_depth, third_depth], 
        stddev=0.01), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, third_depth, last_depth], 
        stddev=0.01), name='wc4'),
    # Fully connected layers
    'wd1': tf.Variable(tf.random_normal([out_conv_size, hidden], stddev=0.01), name='wd1'),
    'wd2': tf.Variable(tf.random_normal([hidden, second_hidden], stddev=0.01), name='wd2'),
    'out': tf.Variable(tf.random_normal([second_hidden, n_classes], stddev=0.01), name='wout')
}

biases = {
    # Convolution layers
    'bc1': tf.Variable(tf.zeros([depth]), name='bc1'),
    'bc2': tf.Variable(tf.constant(1.0, shape=[second_depth]), name='bc2'),
    'bc3': tf.Variable(tf.zeros([third_depth]), name='bc3'),
    'bc4': tf.Variable(tf.constant(1.0, shape=[last_depth]), name='bc4'),
    # Fully connected layers
    'bd1': tf.Variable(tf.constant(1.0, shape=[hidden]), name='bd1'),
    'bd2': tf.Variable(tf.constant(1.0, shape=[second_hidden]), name='bd2'),
    'out': tf.Variable(tf.constant(1.0, shape=[n_classes]), name='bout')
}

pool_strides = {
    'first_pool_layer': [1, first_pool_stride_z, first_pool_stride_x, first_pool_stride_y, 1],
    'second_pool_layer': [1, pool_stride, pool_stride, pool_stride, 1],
    'third_pool_layer': [1, pool_stride, pool_stride, pool_stride, 1],
}

pool_windows = {
    'first_pool_layer': [1, first_pool_window_z, first_pool_window_x, first_pool_window_y, 1],
    'second_pool_layer': [1, pool_window, pool_window, pool_window, 1],
    'third_pool_layer': [1, pool_window, pool_window, pool_window, 1]
}