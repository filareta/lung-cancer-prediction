import tensorflow as tf
from functools import reduce

from model_definition import pool_windows, pool_strides


# Create some wrappers for simplicity
def conv3d(x, W, b, name, strides=[1, 1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name) as scope:
        # Conv3D wrapper, with bias and relu activation
        x = tf.nn.conv3d(x, W, strides=strides, padding=padding, name=scope.name)
        x = tf.nn.bias_add(x, b, name=scope.name)
        return tf.nn.relu(x, name=scope.name)


def maxpool3d(x, name, k, strides=[1, 1, 1, 1, 1], padding='SAME'):
    # MaxPool3D wrapper
    return tf.nn.max_pool3d(x, ksize=k, strides=strides, padding=padding, name=name)


def fc(x, weights, bias, name, dropout=None, with_relu=True):
    with tf.variable_scope(name) as scope:
        fc = tf.add(tf.matmul(x, weights), bias, name=scope.name)
        if with_relu:
            fc = tf.nn.relu(fc, name=scope.name)
        if dropout:
            fc = tf.nn.dropout(fc, dropout)

        return fc


# Create model
def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], name="conv1", 
        strides=[1, 2, 2, 2, 1], padding='VALID')
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, name="pool1",
                      k=pool_windows['first_pool_layer'],
                      strides=pool_strides['first_pool_layer'],
                      padding='VALID')
    print("After first layer: ", conv1.get_shape().as_list())

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'], name="conv2",
        padding='VALID')
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, name="pool2", k=pool_windows['second_pool_layer'], 
                      strides=pool_strides['second_pool_layer'],
                      padding='VALID')
    print("After second layer: ", conv2.get_shape().as_list())

    # Convolution Layer
    conv3 = conv3d(conv2, weights['wc3'], biases['bc3'], name="conv3",
        padding='VALID')
    print("After third layer: ", conv3.get_shape().as_list())
    
    # Convolution Layer
    conv4 = conv3d(conv3, weights['wc4'], biases['bc4'], name="conv4",
        padding='VALID')
    # Max Pooling (down-sampling)
    conv4 = maxpool3d(conv4, name="pool3", k=pool_windows['third_pool_layer'], 
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
    fc1 = fc(fc1, weights['wd1'], biases['bd1'], name='first_fully_connected', 
        dropout=dropout)

    fc2 = fc(fc1, weights['wd2'], biases['bd2'], name='second_fully_connected', 
        dropout=dropout)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'], name='output_layer')
    return out


def loss_function_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)


# Sparse sofmtax is used for mutually exclusive classes,
# labels rank must be logits rank - 1
def sparse_loss_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)
