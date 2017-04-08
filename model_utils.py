import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import tensorflow as tf

import config
from utils import store_to_csv, read_csv


def store_error_plots(validation_err, train_err):
    try:
        plt.plot(validation_err)
        plt.savefig("validation_errors.png")

        plt.plot(train_err)
        plt.savefig("train_errors.png")
    except Exception as e:
        print("Drawing errors failed with: {}".format(e))


# TODO: Think of a better evaluation strategy
def high_error_increase(errors, 
                        current, 
                        least_count=3, 
                        incr_threshold=0.1):
    if len(errors) < least_count:
        return False

    return any(current - x >= incr_threshold 
        for x in errors)


def accuracy(predictions, labels):
    return (100 * np.sum(np.argmax(predictions, 1) == labels) 
        / predictions.shape[0])


def evaluate_log_loss(predictions, labels):
    return log_loss(labels, predictions, labels=[0, 1])


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
        batch_pred = sess.run(valid_prediction, 
            feed_dict={feed_data_key: np.stack(validation_batch)})
       
        validation_pred.extend(batch_pred)
        validation_labels.extend(validation_label)
        index += batch_size

    validation_acc = accuracy(np.stack(validation_pred), 
        np.stack(validation_labels))
    validation_log_loss = evaluate_log_loss(validation_pred, 
                                            validation_labels)

    return (validation_acc, validation_log_loss)


def evaluate_test_set(sess, 
                      test_set,
                      test_img_shape,
                      test_prediction,
                      feed_data_key,
                      export_csv=True):
    i = 0
    patients, probs = [], []

    try:
        while i < test_set.num_samples:
            patient, test_img = test_set.next_patient()
            test_img_reshape = tf.reshape(test_img, 
                shape=test_img_shape)
            test_img = sess.run(test_img_reshape)
            i += 1
            # returns index of column with highest probability
            # [first class=no cancer=0, second class=cancer=1]
            if len(test_img):
                output = sess.run(test_prediction, 
                    feed_dict={feed_data_key: test_img})
                max_ind_f = tf.argmax(output, 1)
                ind_value = sess.run(max_ind_f)
                patients.append(patient)
                max_prob = output[0][ind_value][0]
                if ind_value[0] == config.NO_CANCER_CLS:
                    max_prob = 1.0 - max_prob
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

    print("Log loss: ", round(log_loss_err, 5))
    print("Accuracy: %.1f%%" % acc)

    if with_merged_report:
        df = pd.DataFrame(data={'prediction': probs, 'label': labels},
                     columns=['prediction', 'label'],
                     index=true_labels.index)
        df.to_csv('report_{}'.format(os.path.basename(sample_solution)))

    return (log_loss_err, acc)


