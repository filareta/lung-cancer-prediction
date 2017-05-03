import tensorflow as tf
from functools import reduce

from model_definition import pool_windows, pool_strides


# Create some wrappers for simplicity
def conv3d(x, W, b, strides=[1, 1, 1, 1, 1], padding='SAME'):
    # Conv3D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=strides, padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool3d(x, k, strides=[1, 1, 1, 1, 1], padding='SAME'):
    # MaxPool3D wrapper
    return tf.nn.max_pool3d(x, ksize=k, strides=strides, padding=padding)


# Create model
def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], 
        strides=[1, 2, 2, 2, 1], padding='VALID')
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=pool_windows['first_pool_layer'], 
                      strides=pool_strides['first_pool_layer'],
                      padding='VALID')
    print("After first layer: ", conv1.get_shape().as_list())

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'], padding='VALID')
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=pool_windows['second_pool_layer'], 
                      strides=pool_strides['second_pool_layer'],
                      padding='VALID')
    print("After second layer: ", conv2.get_shape().as_list())

    # Convolution Layer
    conv3 = conv3d(conv2, weights['wc3'], biases['bc3'], padding='VALID')
    print("After third layer: ", conv3.get_shape().as_list())
    
    # Convolution Layer
    conv4 = conv3d(conv3, weights['wc4'], biases['bc4'], padding='VALID')
    # Max Pooling (down-sampling)
    conv4 = maxpool3d(conv4, k=pool_windows['third_pool_layer'], 
                      strides=pool_strides['third_pool_layer'],
                      padding='VALID')
    conv4 = tf.nn.dropout(conv4, dropout)
    
    conv_shape = conv4.get_shape().as_list()
    fully_con_input_size = reduce(lambda x, y: x * y, conv_shape[1:])
    print("SHAPE of the last convolution layer after max pooling: {}, new shape {}".format(
        conv_shape, fully_con_input_size))

    # Fully connected layer
    # Reshape conv output to fit fully connected layer input
    number = conv_shape[0] or -1
    fc1 = tf.reshape(conv4, [number, fully_con_input_size])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


def loss_function_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)


# Sparse sofmtax is used for mutually exclusive classes,
# labels rank must be logits rank - 1
def sparse_loss_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)
