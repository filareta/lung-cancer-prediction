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
from model_utils import get_specifity, get_sensitivity
from model import conv_net, loss_function_with_logits, sparse_loss_with_logits


training_iters = 101
save_step = 10
display_steps = 20
validaton_logg_loss_incr_threshold = 0.05
last_errors = 2
tolerance = 15


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

with tf.name_scope("cross_entropy"):
    # Define loss and optimizer
    cost = sparse_loss_with_logits(pred, y)
    # add l2 regularization on the weights on the fully connected layer
    regularizer = tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['wd2'])
    cost = tf.reduce_mean(cost + beta * regularizer)

trainable_vars = tf.trainable_variables()

with tf.name_scope("train"):
    gradients = tf.gradients(cost, trainable_vars)
    gradients = list(zip(gradients, trainable_vars))
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in trainable_vars:
  tf.summary.histogram(var.name, var)


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


merged = tf.summary.merge_all()

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
best_validation_sensitivity = 0.0

# Add summary for log loss per epoch, accuracy and sensitivity
with tf.name_scope("log_loss"):
    log_loss = tf.placeholder(tf.float32, name="log_loss_per_epoch")
    
loss_summary = tf.summary.scalar("log_loss", log_loss)

with tf.name_scope("sensitivity"):
    sensitivity = tf.placeholder(tf.float32, name="sensitivity_per_epoch")

sensitivity_summary = tf.summary.scalar("sensitivity", sensitivity)

with tf.name_scope("accuracy"):
    tf_accuracy = tf.placeholder(tf.float32, name="accuracy_per_epoch")

accuracy_summary = tf.summary.scalar("accuracy", tf_accuracy)


def export_evaluation_summary(log_loss_value, 
                              accuracy_value, 
                              sensitivity_value, 
                              step,
                              sess,
                              writer):
    error_summary, acc_summary, sens_summary = sess.run(
        [loss_summary, accuracy_summary, sensitivity_summary],
        feed_dict={log_loss: log_loss_value, tf_accuracy: accuracy_value, 
                   sensitivity: sensitivity_value})
    writer.add_summary(error_summary, global_step=step)
    writer.add_summary(acc_summary, global_step=step)
    writer.add_summary(sens_summary, global_step=step)
    writer.flush()


# Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(config.SUMMARIES_DIR + '/train')
    validation_writer = tf.summary.FileWriter(config.SUMMARIES_DIR + '/validation')

    if config.RESTORE:
        print("Restoring model from last saved state: ", config.RESTORE_MODEL_CKPT)
        saver.restore(sess, model_out_dir + config.RESTORE_MODEL_CKPT)
    
    sess.run(init)

    # Add the model graph to TensorBoard
    if not config.RESTORE:
        train_writer.add_graph(sess.graph)

    for step in range(config.START_STEP, training_iters):
        train_pred = []
        train_labels = []

        for i in range(training_set.num_samples):
            batch_data, batch_labels = training_set.next_batch(batch_size)
            reshaped = sess.run(reshape_op, feed_dict={input_img: np.stack(batch_data)})
            feed_dict = {x: reshaped, y: batch_labels, keep_prob: dropout}

            if step % display_steps == 0:
                _, loss, predictions, summary = sess.run([train_op, cost, train_prediction, merged], 
                                                          feed_dict=feed_dict)

                try:
                    train_writer.add_summary(summary, step + i)
                except Exception as e:
                    print("Exeption raised during summary export. ", e)
            else:
                _, loss, predictions = sess.run([train_op, cost, train_prediction], 
                                                 feed_dict=feed_dict)

            train_pred.extend(predictions)
            train_labels.extend(batch_labels)

        train_writer.flush()

        if step % save_step == 0:
            print("Storing model snaphost...")
            saver.save(sess, model_store_path(model_out_dir, 'lungs' + str(step)))

        
        print("============== Train Epoch {} finished! {} samples processed.".format(
            training_set.finished_epochs, len(train_pred)))
        train_acc_epoch = accuracy(np.stack(train_pred), np.stack(train_labels))

        train_log_loss = evaluate_log_loss(train_pred, train_labels)
    
        print('<-===== Train log loss error {} ======->'.format(train_log_loss))
        print('<-===== Train set accuracy {} ======->'.format(train_acc_epoch))
        print('================ Train set confusion matrix ====================')
        confusion_matrix = display_confusion_matrix_info(train_labels, train_pred)
        train_sensitivity = get_sensitivity(confusion_matrix)
        train_specifity = get_specifity(confusion_matrix)
        print('Test data sensitivity {} and specifity {}'.format(train_sensitivity, train_specifity))

        export_evaluation_summary(train_log_loss, 
                                  train_acc_epoch, 
                                  train_sensitivity, 
                                  step, sess, train_writer)

        print('<<<<<<<<<<Evaluate validation set>>>>>>>>>>>>>>>>')
        validation_acc, validation_log_loss, val_sensitivity, val_specifity = evaluate_validation_set(sess, 
                                                                                                      validation_set,
                                                                                                      valid_prediction,
                                                                                                      tf_valid_dataset,
                                                                                                      batch_size)
        export_evaluation_summary(validation_log_loss, 
                                  validation_acc, 
                                  val_sensitivity, 
                                  step, sess, validation_writer)

        print('Validation accuracy: %.1f%%' % validation_acc)
        print('<<=== LOG LOSS overall validation samples: {} ===>>.'.format(
            validation_log_loss))
        print('Validation set sensitivity {} and specifity {}'.format(val_sensitivity, val_specifity))

        if validation_log_loss < best_validation_err and val_sensitivity > best_validation_sensitivity:
            best_validation_err = validation_log_loss
            best_validation_sensitivity = val_sensitivity
            print("Storing model snaphost with best validation error {} and sensitivity {} ".format(
                best_validation_err, best_validation_sensitivity))
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

    train_writer.close()
    validation_writer.close()

    saver.save(sess, model_store_path(model_out_dir, 'last'))
    print("Model saved...")
    store_error_plots(validation_errors, train_errors_per_epoch)


    # ============= REAL TEST DATA EVALUATION =====================
    evaluate_test_set(sess,
                      exact_tests,
                      test_prediction,
                      tf_test_dataset)
