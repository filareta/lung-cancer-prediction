import pandas as pd
import numpy as np

import tensorflow as tf
import data_set as ds
import config

from model_definition import x, y, keep_prob, learning_rate, batch_size
from model_definition import tf_valid_dataset, tf_test_dataset
from model_definition import weights, biases, dropout

from model_utils import evaluate_log_loss, accuracy, store_error_plots
from model import conv_net


training_iters = 101
save_step = 5


# Add tensors to collection stored in the model graph
# definition
tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)
tf.add_to_collection('vars', keep_prob)
tf.add_to_collection('vars', tf_valid_dataset)
tf.add_to_collection('vars', tf_test_dataset)

for weight_key, weigth_var in weights.items():
    tf.add_to_collection('vars', weigth_var)

for bias_key, bias_var in biases.items():
    tf.add_to_collection('vars', bias_var)


# Construct model
pred = conv_net(x, weights, biases, dropout)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), 
    name='cost-func')
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

# ======= Training ========

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
            print("Low enough log loss validation error, terminate!")
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
            # [first class=no cancer=0, second class=cancer=1]
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

