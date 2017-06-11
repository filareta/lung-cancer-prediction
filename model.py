import tensorflow as tf
from functools import reduce

from model_configuration import DefaultConfig


class Convolution3DNetwork(object):
    DEFAULT_LAYER_PADDING = 'VALID'
    DEFAULT_CONV_STRIDE = [1, 1, 1, 1, 1]

    def __init__(self, config=None):
        self._config = config or DefaultConfig()
        self._strides = self._config.get_strides()
        self._pool_strides = self._config.get_pool_strides()
        self._pool_windows = self._config.get_pool_windows()

        self._init_weights()
        self._init_biases()


    def _init_weights(self):
        self._weights = [
            tf.Variable(init_func, name=name) 
            for name, init_func in self._config.get_fc_weights()
            ]
        self._conv_weights = [
                tf.Variable(init_func, name=name) 
                for name, init_func in self._config.get_conv_weights()
            ]

    def _init_biases(self):
        self._biases = [
                tf.Variable(init_func, name=name) 
                for name, init_func in self._config.get_fc_biases()
            ]
        self._conv_biases = [
                tf.Variable(init_func, name=name) 
                for name, init_func in self._config.get_conv_biases()
            ]

    def weights(self):
        self._conv_weights + self._weights

    def l2_regularizer(self):
        if self._config.with_l2_norm():
            return reduce(lambda x, y: tf.nn.l2_loss(x) + tf.nn.l2_loss(y),
                          self._weights)
        return 0

    def biases(self):
        return self._conv_biases + self._biases

    # Create some wrappers for simplicity
    def conv3d(self, x, W, b, name, 
               strides=DEFAULT_CONV_STRIDE,
               padding=DEFAULT_LAYER_PADDING):
        with tf.variable_scope(name) as scope:
            # Conv3D wrapper, with bias and relu activation
            x = tf.nn.conv3d(x, W, strides=strides, 
                             padding=padding, name=scope.name)
            x = tf.nn.bias_add(x, b, name='bias')
            return tf.nn.relu(x, name='relu')

    def maxpool3d(self, x, name, k, 
                  strides=DEFAULT_CONV_STRIDE, 
                  padding=DEFAULT_LAYER_PADDING):
        # MaxPool3D wrapper
        return tf.nn.max_pool3d(x, ksize=k, strides=strides, 
                                padding=padding, name=name)

    def fc(self, x, weights, bias, name, dropout=None, with_relu=True):
        with tf.variable_scope(name) as scope:
            fc = tf.add(tf.matmul(x, weights), bias, name=scope.name)
            if with_relu:
                fc = tf.nn.relu(fc, name='relu')
            if dropout:
                fc = tf.nn.dropout(fc, dropout)

            return fc

    # Create model
    def conv_net(self, x, dropout):
        # Convolution Layer
        last_conv_layer = x

        for i, weight in enumerate(self._conv_weights):
             # Convolution Layer
            last_conv_layer = self.conv3d(last_conv_layer, weight, 
                                         self._conv_biases[i],
                                         name="conv" + str(i), 
                                         strides=self._strides[i])
          
            # Max Pooling (down-sampling)
            if self._pool_windows[i]:
                last_conv_layer = self.maxpool3d(last_conv_layer, 
                                                name="pool" + str(i),
                                                k=self._pool_windows[i],
                                                strides=self._pool_strides[i])

            print("After current layer: ", last_conv_layer.get_shape().as_list())
        
        
        if self._config.has_dropout_after_convolutions():
            last_conv_layer = tf.nn.dropout(last_conv_layer, dropout)

        conv_shape = last_conv_layer.get_shape().as_list()
        fully_con_input_size = reduce(lambda x, y: x * y, conv_shape[1:])
        print("SHAPE of the last convolution layer after max pooling: {}, new shape {}".format(
            conv_shape, fully_con_input_size))

        # Fully connected layer
        # Reshape conv output to fit fully connected layer input
        number = conv_shape[0] or -1
        fully_connected = tf.reshape(last_conv_layer, [number, fully_con_input_size])

        for i, weight in enumerate(self._weights[:-1]):
            if self._config.has_fc_dropout(i):
                layer_dropout = dropout
            else:
                layer_dropout = None

            fully_connected = self.fc(fully_connected, 
                                     weight, 
                                     self._biases[i], 
                                     name='fully_connected' + str(i),
                                     dropout=layer_dropout)
        
        # Output, class prediction
        out = tf.add(tf.matmul(fully_connected, self._weights[-1]),
                     self._biases[-1], name='output_layer')
        return out


def loss_function_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)


# Sparse sofmtax is used for mutually exclusive classes,
# labels rank must be logits rank - 1
def sparse_loss_with_logits(logits, labels, tensor_name='cost_func'):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels), name=tensor_name)
