import pandas as pd
import numpy as np

import tensorflow as tf
import data_set as ds
import config

from model_definition import x, y, keep_prob, learning_rate, batch_size
from model_definition import tf_valid_dataset, tf_test_dataset
from model_definition import weights, biases, dropout
from model_utils import input_img, reshape_op

from model_utils import evaluate_log_loss, accuracy, evaluate_validation_set
from model_utils import model_store_path, store_error_plots, evaluate_test_set
from model_utils import high_error_increase, display_confusion_matrix_info
from model import conv_net, loss_function_with_logits, sparse_loss_with_logits


training_iters = 101
save_step = 5
validaton_logg_loss_incr_threshold = 0.05
last_errors = 2
tolerance = 5


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

beta = 0.01
# Define loss and optimizer
cost = sparse_loss_with_logits(pred, y)
# add l2 regularization on the weights on the fully connected layer
regularizer = tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['wd2'])
cost = tf.reduce_mean(cost + beta * regularizer)
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
model_out_dir = data_loader.results_out_dir()

print('Validation examples count: ', validation_set.num_samples)
print('Test examples count: ', exact_tests.num_samples)
print('Model will be stored in: ', model_out_dir)


# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

validation_errors = []
train_errors_per_epoch = []
best_validation_err = 1.0


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for step in range(1, training_iters):
        last_epoch = training_set.finished_epochs
        train_pred = []
        train_labels = []

        while last_epoch == training_set.finished_epochs:
            batch_data, batch_labels = training_set.next_batch(batch_size)
            reshaped = sess.run(reshape_op, feed_dict={input_img: np.stack(batch_data)})
            feed_dict = {x: reshaped, y: batch_labels, keep_prob: dropout}
            _, loss, predictions = sess.run([optimizer, cost, train_prediction], 
                                            feed_dict=feed_dict)
            train_pred.extend(predictions)
            train_labels.extend(batch_labels)

        if step % save_step == 0:
            print("Storing model snaphost...")
            saver.save(sess, model_store_path(model_out_dir, 'lungs' + str(step)))

        
        print("============== Train Epoch {} finished!================".format(
            training_set.finished_epochs))
        train_acc_epoch = accuracy(np.stack(train_pred), np.stack(train_labels))

        train_log_loss = evaluate_log_loss(train_pred, train_labels)
        print('<-===== Train log loss error {} ======->'.format(train_log_loss))
        print("================ Train set confusion matrix ====================")
        display_confusion_matrix_info(train_labels, train_pred)

        

        print("<<<<<<<<<<Evaluate validation set>>>>>>>>>>>>>>>>")
        validation_acc, validation_log_loss = evaluate_validation_set(sess, 
                                                                      validation_set,
                                                                      valid_prediction,
                                                                      tf_valid_dataset,
                                                                      batch_size)
        
        print('Validation accuracy: %.1f%%' % validation_acc)
        print("<<=== LOG LOSS overall validation samples: {} ===>>.".format(
            validation_log_loss))

        if validation_log_loss < best_validation_err:
            best_validation_err = validation_log_loss
            print("Storing model snaphost with best validation error: ", best_validation_err)
            if step % save_step != 0:
                saver.save(sess, model_store_path(model_out_dir, 'best_err' + str(step)))

        if validation_log_loss < 0.1:
            print("Low enough log loss validation error, terminate!")
            break;

        if high_error_increase(validation_errors[-last_errors:], 
                               validation_log_loss,
                               last_errors,
                               validaton_logg_loss_incr_threshold):
            if tolerance and train_log_loss <= train_errors_per_epoch[-1]:
                print("Train error still decreases, continue...")
                tolerance -= 1
                validation_errors.append(validation_log_loss)
                train_errors_per_epoch.append(train_log_loss)
                continue

            print("Validation log loss has increased more than the allowed threshold",
                  " for the past iterations, terminate!")
            print("Last iterations: ", validation_errors[-last_errors:])
            print("Current validation error: ", validation_log_loss)
            break

        validation_errors.append(validation_log_loss)
        train_errors_per_epoch.append(train_log_loss)

    saver.save(sess, model_store_path(model_out_dir, 'last'))
    print("Model saved...")
    store_error_plots(validation_errors, train_errors_per_epoch)


    # ============= REAL TEST DATA EVALUATION =====================
    evaluate_test_set(sess,
                      exact_tests,
                      test_prediction,
                      tf_test_dataset)
