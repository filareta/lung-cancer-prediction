import tensorflow as tf

import model_definition as md


# Create some wrappers for simplicity
def conv3d(x, W, b, strides=1, padding='SAME'):
    # Conv3D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool3d(x, k=[1, md.pool_window, md.pool_window, md.pool_window, 1],
              strides=[1, md.pool_stride, md.pool_stride, md.pool_stride, 1],
              padding='SAME'):
    # MaxPool3D wrapper
    return tf.nn.max_pool3d(x, ksize=k, strides=strides, padding=padding)


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=md.image_tensor_shape)

    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], padding='VALID')
    conv_shape = conv1.get_shape().as_list()
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=[1, md.first_pool_window_z, md.first_pool_window, md.first_pool_window, 1], 
                      strides=[1, md.first_pool_stride_z, md.first_pool_stride, md.first_pool_stride, 1],
                      padding='VALID')

    conv_shape = conv1.get_shape().as_list()
    print("After first: ", conv_shape)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'], padding='VALID')
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, padding='VALID')

    conv_shape = conv2.get_shape().as_list()
    print("After second: ", conv_shape)

    # Convolution Layer
    conv3 = conv3d(conv2, weights['wc3'], biases['bc3'], padding='VALID')

    conv_shape = conv3.get_shape().as_list()
    print("After third: ", conv_shape)
    # Convolution Layer
    conv4 = conv3d(conv3, weights['wc4'], biases['bc4'], padding='VALID')

    conv5 = conv3d(conv4, weights['wc5'], biases['bc5'], padding='VALID')
    # Max Pooling (down-sampling)
    conv5 = maxpool3d(conv5, padding='VALID')
    
    conv_shape = conv5.get_shape().as_list()
    print("SHAPE of the last convolution layer after max pooling: {}, new shape {}".format(
        conv_shape, conv_shape[1]*conv_shape[2]*conv_shape[3]*conv_shape[4]))

    # Fully connected layer
    # Reshape conv output to fit fully connected layer input
    number = conv_shape[0] or -1
    fc1 = tf.reshape(conv5, [number, conv_shape[1]*conv_shape[2]*conv_shape[3]*conv_shape[4]])
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
