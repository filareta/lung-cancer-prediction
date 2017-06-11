import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, confusion_matrix
import tensorflow as tf

import config
from utils import store_to_csv, read_csv

# Network Input Parameters
n_x = config.IMAGE_PXL_SIZE_X
n_y = config.IMAGE_PXL_SIZE_Y
n_z = config.SLICES
num_channels = 1

# tf Graph input
x = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, n_z, n_x, n_y, num_channels), 
    name='train_input')
y = tf.placeholder(tf.int32, shape=(config.BATCH_SIZE,), name='label')
keep_prob = tf.placeholder(tf.float32, name='dropout') #dropout (keep probability)

tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), 
    name='validation_set')
tf_test_dataset = tf.placeholder(tf.float32, shape=(None, n_z, n_x, n_y, num_channels), 
    name='test_set')

input_img = tf.placeholder(tf.float32, 
    shape=(1, config.SLICES, config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y))
# Reshape input picture, first dimension is kept to be able to support batches
reshape_op = tf.reshape(input_img, 
    shape=(-1, config.SLICES, config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y, 1))

input_test_img = tf.placeholder(tf.float32, 
    shape=(config.SLICES, config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y))
# Reshape test input picture
reshape_test_op = tf.reshape(input_test_img, 
    shape=(-1, config.SLICES, config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y, 1))


def store_error_plots(validation_err, train_err):
    try:
        plt.plot(validation_err)
        plt.savefig("validation_errors.png")

        plt.plot(train_err)
        plt.savefig("train_errors.png")
    except Exception as e:
        print("Drawing errors failed with: {}".format(e))


def high_error_increase(errors, 
                        current, 
                        least_count=3, 
                        incr_threshold=0.1):
    if len(errors) < least_count:
        return False

    return any(current - x >= incr_threshold 
        for x in errors)


def get_max_prob(output, ind_value):
    max_prob = output[ind_value]
    if ind_value == config.NO_CANCER_CLS:
        max_prob = 1.0 - max_prob

    return max_prob


def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == labels) 
        / predictions.shape[0])


def evaluate_log_loss(predictions, target_labels):
    return log_loss(target_labels, predictions, labels=[0, 1])


def get_confusion_matrix(target_labels, predictions, labels=[0, 1]):
    predicted_labels = np.argmax(predictions, 1)
    return confusion_matrix(target_labels, predicted_labels, labels)


def display_confusion_matrix_info(target_labels, predictions, labels=[0, 1]):
    matrix = get_confusion_matrix(target_labels, predictions, labels)
    print("True negatives count: ", matrix[0][0])
    print("False negatives count: ", matrix[1][0])
    print("True positives count: ", matrix[1][1])
    print("False positives count: ", matrix[0][1])

    return matrix

def get_sensitivity(confusion_matrix):
    true_positives = confusion_matrix[1][1]
    false_negatives = confusion_matrix[1][0]

    return true_positives / float(true_positives + false_negatives)


def get_specifity(confusion_matrix):
    true_negatives = confusion_matrix[0][0]
    false_positives = confusion_matrix[0][1]

    return true_negatives / float(true_negatives + false_positives)


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


def model_store_path(store_dir, step):
    return os.path.join(store_dir, 
        'model_{}.ckpt'.format(step))


def evaluate_validation_set(sess, 
                            validation_set, 
                            valid_prediction, 
                            feed_data_key, 
                            batch_size):
    validation_pred = []
    validation_labels = []

    index = 0
    while index < validation_set.num_samples:
        validation_batch, validation_label = validation_set.next_batch(batch_size)
        reshaped = sess.run(reshape_op, feed_dict={input_img: np.stack(validation_batch)})
        batch_pred = sess.run(valid_prediction, 
            feed_dict={feed_data_key: reshaped})
       
        validation_pred.extend(batch_pred)
        validation_labels.extend(validation_label)
        index += batch_size

    validation_acc = accuracy(np.stack(validation_pred), 
        np.stack(validation_labels))
    validation_log_loss = evaluate_log_loss(validation_pred, 
                                            validation_labels)

    confusion_matrix = display_confusion_matrix_info(validation_labels, validation_pred)
    sensitivity = get_sensitivity(confusion_matrix)
    specifity = get_specifity(confusion_matrix)

    return (validation_acc, validation_log_loss, sensitivity, specifity)


def evaluate_test_set(sess, 
                      test_set,
                      test_prediction,
                      feed_data_key,
                      export_csv=True):
    i = 0
    patients, probs = [], []

    try:
        while i < test_set.num_samples:
            patient, test_img = test_set.next_patient()
            test_img = sess.run(reshape_test_op, feed_dict={input_test_img: test_img})
            i += 1
            # returns index of column with highest probability
            # [first class=no cancer=0, second class=cancer=1]
            if len(test_img):
                patients.append(patient)
                output = sess.run(test_prediction, 
                    feed_dict={feed_data_key: test_img})
                max_ind_f = tf.argmax(output, 1)
                ind_value = sess.run(max_ind_f)
                max_prob = get_max_prob(output[0], ind_value[0])
                probs.append(max_prob)

                print("Output {} for patient with id {}, predicted output {}.".format(
                    max_prob, patient, output[0]))

            else:
                print("Corrupted test image, incorrect shape for patient {}".format(
                    patient))
    except Exception as e:
        print("Storing results failed with: {}".format(e))

    if export_csv:
        store_to_csv(patients, probs, config.SOLUTION_FILE_PATH)


def evaluate_solution(sample_solution, with_merged_report=True):
    true_labels = read_csv(config.REAL_SOLUTION_CSV)
    predictions = read_csv(sample_solution)
    patients = true_labels.index.values

    probs, labels, probs_cls = [], [], []
    for patient in patients:
        prob = predictions.get_value(patient, config.COLUMN_NAME)
        probs.append(prob)
        probs_cls.append([1.0 - prob, prob])
        labels.append(true_labels.get_value(patient, config.COLUMN_NAME))
    
    probs_cls = np.array(probs_cls)
    log_loss_err = evaluate_log_loss(probs_cls, labels)
    acc = accuracy(probs_cls, np.array(labels))

    confusion_matrix = display_confusion_matrix_info(labels, probs_cls)
    sensitivity = get_sensitivity(confusion_matrix)
    specifity = get_specifity(confusion_matrix)

    print("Log loss: ", round(log_loss_err, 5))
    print("Accuracy: %.1f%%" % acc)
    print("Sensitivity: ", round(sensitivity, 5))
    print("Specifity: ", round(specifity, 5))

    if with_merged_report:
        df = pd.DataFrame(data={'prediction': probs, 'label': labels},
                          columns=['prediction', 'label'],
                          index=true_labels.index)
        df.to_csv('report_{}'.format(os.path.basename(sample_solution)))

    return (log_loss_err, acc, sensitivity, specifity)


