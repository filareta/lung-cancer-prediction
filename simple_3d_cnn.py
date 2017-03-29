from __future__ import print_function
from itertools import repeat
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np

import tensorflow as tf
import data_set as ds
import input_dicoms as ind


# Parameters
learning_rate = 0.001
training_iters = 101
training_epoches = 100
batch_size = 4
save_step = 5
display_step = 50
depth = 16
second_depth = 32
third_depth = 64
last_depth = 32
hidden = 100
second_hidden = 50
num_channels = 1
kernel_size = 3
first_kernel_size = 11
first_kernel_size_z = 9
first_pool_stride = 4
first_pool_window = 5
first_pool_stride_z = 2
first_pool_window_z = 3
pool_window = 3
pool_stride = 2

# Network Parameters
n_x = ind.IMAGE_PXL_SIZE
n_y = ind.IMAGE_PXL_SIZE
n_z = ind.HM_SLICES
n_input = n_x * n_y * n_z
n_classes = 2
dropout = 0.8 # Dropout, probability to keep units
validaton_logg_loss_incr_threshold = 0.3
last_errors = 3


def store_error_plots(validation_err, train_err):
    try:
        plt.plot(validation_err)
        plt.savefig("validation_errors.png")

        plt.plot(train_err)
        plt.savefig("train_errors.png")
    except Exception as e:
        print("Drawing errors failed with: {}".format(e))


# TODO: Think of a better evaluation strategy
def high_error_increase(errors, current, least_count=last_errors):
    if len(errors) < least_count:
        return False

    return any(current - x >= validaton_logg_loss_incr_threshold 
        for x in errors)


def reformat(data_input):
    return data_input.reshape(n_z, n_x, n_y, num_channels)


def reformat_batch(data_batch):
    return np.stack([reformat(data) for data in data_batch])


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
        / predictions.shape[0])


def evaluate_log_loss(predictions, labels):
    true_labels = np.argmax(labels, 1)

    return log_loss(true_labels, predictions, labels=[0, 1])


def calculate_conv_output_size(x, y, z, strides, filters, paddings):
    # Currently axes are transposed [z, x, y]
    for i, stride in enumerate(strides):
        if paddings[i] == 'VALID':
            print("VALID padding")
            f = filters[i]
            x = np.ceil(np.float((x - f[1] + 1) / float(stride[1])))
            y = np.ceil(np.float((y - f[2] + 1) / float(stride[2])))
            z = np.ceil(np.float((z - f[0] + 1) / float(stride[0])))
            print("Calculating X: {}, Y: {}, Z: {}.".format(x, y, z))
        else:
            print("SAME padding")
            x = np.ceil(float(x) / float(stride[1]))
            y = np.ceil(float(y) / float(stride[2]))
            z = np.ceil(float(z) / float(stride[0]))
    print("Final X: {}, Y: {}, Z: {}.".format(x, y, z))

    return (x, y, z)


# tf Graph input
x = tf.placeholder(tf.float32, shape=(batch_size, n_z, n_x, n_y, num_channels), name='train_input')
y = tf.placeholder(tf.float32, shape=(batch_size, n_classes), name='label')
keep_prob = tf.placeholder(tf.float32, name='dropout') #dropout (keep probability)

tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), name='validation_set')
tf_test_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), name='test_set')

tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)
tf.add_to_collection('vars', keep_prob)
tf.add_to_collection('vars', tf_valid_dataset)
tf.add_to_collection('vars', tf_test_dataset)


# Create some wrappers for simplicity
def conv3d(x, W, b, strides=1, padding='SAME'):
    # Conv3D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# TODO: Check proper padding here
def maxpool3d(x, k=[1, pool_window, pool_window, pool_window, 1],
              strides=[1, pool_stride, pool_stride, pool_stride, 1],
              padding='SAME'):
    # MaxPool3D wrapper
    return tf.nn.max_pool3d(x, ksize=k, strides=strides, padding=padding)


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, n_z, n_x, n_y, num_channels])

    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], padding='VALID')
    conv_shape = conv1.get_shape().as_list()
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=[1, first_pool_window_z, first_pool_window, first_pool_window, 1], 
                      strides=[1, first_pool_stride_z, first_pool_stride, first_pool_stride, 1],
                      padding='VALID')
    # conv1 = maxpool3d(conv1)
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
    # Max Pooling (down-sampling)
    # conv3 = maxpool3d(conv3)

    conv_shape = conv3.get_shape().as_list()
    print("After third: ", conv_shape)
    # Convolution Layer
    conv4 = conv3d(conv3, weights['wc4'], biases['bc4'], padding='VALID')
    # # Max Pooling (down-sampling)
    # conv4 = maxpool3d(conv4, padding='VALID')

    conv5 = conv3d(conv4, weights['wc5'], biases['bc5'], padding='VALID')
    # # Max Pooling (down-sampling)
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

# This handles padding in both convolution and pooling layers
strides = [[1, 1, 1],
           [first_pool_stride_z, first_pool_stride, first_pool_stride],
           [1, 1, 1],
           [pool_stride, pool_stride, pool_stride],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [pool_stride, pool_stride, pool_stride]]

filters = [[first_kernel_size_z, first_kernel_size, first_kernel_size],
            [first_pool_window_z, first_pool_window, first_pool_window],
            [kernel_size, kernel_size, kernel_size],
            [pool_window, pool_window, pool_window],
            [kernel_size, kernel_size, kernel_size],
            [kernel_size, kernel_size, kernel_size],
            [kernel_size, kernel_size, kernel_size],
            [pool_window, pool_window, pool_window]]
padding_types = ['VALID', 'VALID', 'VALID', 
                'VALID', 'VALID', 'VALID', 
                'VALID', 'VALID',]
new_x, new_y, new_z = calculate_conv_output_size(n_x, n_y, n_z, 
                                                 strides, 
                                                 filters,
                                                 padding_types)

out_conv_size = int(new_x * new_y * new_z * last_depth)
print("Last conv net output size should be {}".format(out_conv_size))


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([first_kernel_size_z, first_kernel_size, first_kernel_size, num_channels, depth], stddev=0.01), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, depth, second_depth], stddev=0.01), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, second_depth, third_depth], stddev=0.01), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, third_depth, third_depth], stddev=0.01), name='wc4'),
    'wc5': tf.Variable(tf.random_normal([kernel_size, kernel_size, kernel_size, third_depth, last_depth], stddev=0.01), name='wc5'),
    'wd1': tf.Variable(tf.random_normal([out_conv_size, hidden], stddev=0.01), name='wd1'),
    'wd2': tf.Variable(tf.random_normal([hidden, second_hidden], stddev=0.01), name='wd2'),
    'out': tf.Variable(tf.random_normal([second_hidden, n_classes], stddev=0.01), name='wout')
}

biases = {
    'bc1': tf.Variable(tf.zeros([depth]), name='bc1'),
    'bc2': tf.Variable(tf.constant(1.0, shape=[second_depth]), name='bc2'),
    'bc3': tf.Variable(tf.zeros([third_depth]), name='bc3'),
    'bc4': tf.Variable(tf.constant(1.0, shape=[third_depth]), name='bc4'),
    'bc5': tf.Variable(tf.constant(1.0, shape=[last_depth]), name='bc5'),
    'bd1': tf.Variable(tf.constant(1.0, shape=[hidden]), name='bd1'),
    'bd2': tf.Variable(tf.constant(1.0, shape=[second_hidden]), name='bd2'),
    'out': tf.Variable(tf.constant(1.0, shape=[n_classes]), name='bout')
}

for weight_key, weigth_var in weights.items():
    tf.add_to_collection('vars', weigth_var)

for bias_key, bias_var in biases.items():
    tf.add_to_collection('vars', bias_var)

# Construct model
pred = conv_net(x, weights, biases, dropout)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='cost-func')
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)


# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(pred, name='train_prediction')
valid_prediction = tf.nn.softmax(conv_net(tf_valid_dataset, weights, biases, 1.0), 
    name='valid_prediction')
test_prediction = tf.nn.softmax(conv_net(tf_test_dataset, weights, biases, 1.0), 
    name='test_prediction')

tf.add_to_collection('vars', cost)
tf.add_to_collection('vars', train_prediction)
tf.add_to_collection('vars', valid_prediction)
tf.add_to_collection('vars', test_prediction)


data_loader = ds.DataLoader()
training_set = data_loader.get_training_set()
validation_set = data_loader.get_validation_set()
exact_tests = data_loader.get_exact_tests_set()


print('Validation examples count: ' + str(validation_set.num_samples))
print('Test examples count: ' + str(exact_tests.num_samples))


# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

validation_errors = []
train_errors_per_epoch = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(1, training_iters): 
        last_epoch = training_set.finished_epochs
        train_errors = []
        train_pred = []
        train_labels = []

        while last_epoch == training_set.finished_epochs:
            batch_data, batch_labels = training_set.next_batch(batch_size)

            feed_dict = {x: np.stack(batch_data), y: batch_labels, keep_prob: dropout}
            _, loss, predictions = sess.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
            train_errors.append(loss)
            train_pred.extend(predictions)
            train_labels.extend(batch_labels)

        if step % save_step == 0:
            print("Store model snaphost!")
            saver.save(sess, './nodules-cl' + str(step) + '.ckpt')

        
        print("============== Train Epoch {} finished!================".format(training_set.finished_epochs))
        train_acc_epoch = accuracy(np.stack(train_pred), np.stack(train_labels))
        mean_err = tf.reduce_mean(train_errors)
        mean_err_value = sess.run(mean_err)
        print('===============Train accuracy %.1f%% on epoch: %d' % (train_acc_epoch, training_set.finished_epochs))
        print('====== Reduced mean error {} ========='.format(mean_err_value))
        train_log_loss = evaluate_log_loss(train_pred, train_labels)
        print('<-=============== Train log loss error {} ==================->'.format(train_log_loss))

        train_errors_per_epoch.append(train_log_loss)

        print("<<<<<<<<<<Evaluate validation set>>>>>>>>>>>>>>>>")
        validation_pred = []
        validation_labels = []

        index = 0
        while index < validation_set.num_samples:
            validation_batch, validation_label = validation_set.next_batch(batch_size)
            batch_pred = sess.run(valid_prediction, feed_dict={tf_valid_dataset: np.stack(validation_batch)})
           
            validation_pred.extend(batch_pred)
            validation_labels.extend(validation_label)
            index += batch_size
        
        validation_acc = accuracy(np.stack(validation_pred), np.stack(validation_labels))
        print('Validation accuracy: %.1f%%' % validation_acc)
        validation_log_loss = evaluate_log_loss(validation_pred, validation_labels)
        print("<<===================LOG LOSS overall validation samples: {}==================>>.".format(validation_log_loss))
        
        if validation_log_loss < 0.1:
            print("Low enought log loss validation error, terminate!")
            break;

        validation_errors.append(validation_log_loss)

    saver.save(sess, './nodules-cl.ckpt')
    print("Model saved!!!")
    store_error_plots(validation_errors, train_errors_per_epoch)


    # #============= REAL EVALUATION =====================
    # Rework
    i = 0
    gen = exact_tests.yield_input()
    patients, outputs, probs = [], [], []
    try:
        while i < data_loader.exact_tests_count:
            patient, test_img = gen.__next__()
            test_img_reshape = tf.reshape(test_img, shape=[-1, n_z, n_x, n_y, num_channels])
            test_img = sess.run(test_img_reshape)
            i += 1
            # returns index of column with highest probability
            #[first class=no cancer=0, second class=cancer=1]
            if len(test_img):
                output = sess.run(test_prediction, feed_dict={tf_test_dataset: test_img})
                max_ind_f = tf.argmax(output, 1)
                ind_value = sess.run(max_ind_f)
                outputs.append(ind_value[0])
                patients.append(patient)
                max_prob = output[0][ind_value][0]
                if ind_value[0] == ds.NO_CANCER_CLS:
                    max_prob = 1.0 - max_prob
                probs.append(max_prob)

                print("Output {} for patient with id {}, max is {}.".format(max_prob, 
                                                                            patient,
                                                                            ind_value[0]))

            else:
                print("Corrupted test image, incorrect shape for patient {}".format(patient))
    except Exception as e:
        print("Storing results failed with: {}".format(e))

    df = pd.DataFrame(data={'id': patients, 'cancer': probs}, columns=['id', 'cancer'], index=None)
    df.to_csv('./sample_solution.csv')

