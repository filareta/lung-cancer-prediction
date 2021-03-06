import os
import pandas as pd
import numpy as np

import tensorflow as tf
import config

from model_utils import x, y, keep_prob
from model_utils import input_img, reshape_op

from model_utils import evaluate_log_loss, accuracy, evaluate_validation_set
from model_utils import model_store_path, store_error_plots, evaluate_test_set
from model_utils import high_error_increase, display_confusion_matrix_info
from model_utils import get_specificity, get_sensitivity, validate_data_loaded
from model import loss_function_with_logits, sparse_loss_with_logits
from model_factory import ModelFactory


# Parameters used during training
batch_size = config.BATCH_SIZE
learning_rate = 0.001
training_iters = 101
save_step = 10
display_steps = 20
validaton_log_loss_incr_threshold = 0.1
last_errors = 2
tolerance = 20
dropout = 0.5 # Dropout, probability to keep units
beta = 0.01

# Construct model
factory = ModelFactory()
model = factory.get_network_model()

if not config.RESTORE:
    # Add tensors to collection stored in the model graph
    # definition
    tf.add_to_collection('vars', x)
    tf.add_to_collection('vars', y)
    tf.add_to_collection('vars', keep_prob)

    for weigth_var in model.weights():
        tf.add_to_collection('vars', weigth_var)

    for bias_var in model.biases():
        tf.add_to_collection('vars', bias_var)

pred = model.conv_net(x, dropout)

with tf.name_scope("cross_entropy"):
    # Define loss and optimizer
    cost = sparse_loss_with_logits(pred, y)
    
    # add l2 regularization on the weights on the fully connected layer
    # if term != 0 is returned
    regularizer = model.l2_regularizer()
    if regularizer != 0:
        print("Adding L2 regularization...")
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
softmax_prediction = tf.nn.softmax(pred, name='softmax_prediction')

if not config.RESTORE:
    tf.add_to_collection('vars', cost)
    tf.add_to_collection('vars', softmax_prediction)


merged = tf.summary.merge_all()

# ======= Training ========
data_loader = factory.get_data_loader()
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
    if not os.path.exists(config.SUMMARIES_DIR):
        os.makedirs(config.SUMMARIES_DIR)
        
    train_writer = tf.summary.FileWriter(os.path.join(config.SUMMARIES_DIR, 'train'))
    validation_writer = tf.summary.FileWriter(os.path.join(config.SUMMARIES_DIR, 
                                                           'validation'))

    sess.run(init)

    if config.RESTORE and \
        os.path.exists(os.path.join(model_out_dir, config.RESTORE_MODEL_CKPT + '.index')):
        
        saver.restore(sess, os.path.join(model_out_dir, config.RESTORE_MODEL_CKPT))
        print("Restoring model from last saved state: ", config.RESTORE_MODEL_CKPT)


    # Add the model graph to TensorBoard
    if not config.RESTORE:
        train_writer.add_graph(sess.graph)

    for step in range(config.START_STEP, training_iters):
        train_pred = []
        train_labels = []

        for i in range(training_set.num_samples):
            batch_data, batch_labels = training_set.next_batch(batch_size)

            if not validate_data_loaded(batch_data, batch_labels):
                break

            reshaped = sess.run(reshape_op, feed_dict={input_img: np.stack(batch_data)})
            feed_dict = {x: reshaped, y: batch_labels, keep_prob: dropout}

            if step % display_steps == 0:
                _, loss, predictions, summary = sess.run([train_op, cost, softmax_prediction, merged], 
                                                          feed_dict=feed_dict)

                try:
                    train_writer.add_summary(summary, step + i)
                except Exception as e:
                    print("Exeption raised during summary export. ", e)
            else:
                _, loss, predictions = sess.run([train_op, cost, softmax_prediction], 
                                                 feed_dict=feed_dict)

            train_pred.extend(predictions)
            train_labels.extend(batch_labels)

        train_writer.flush()

        if step % save_step == 0:
            print("Storing model snaphost...")
            saver.save(sess, model_store_path(model_out_dir, 'lungs' + str(step)))

        
        print("Train epoch {} finished. {} samples processed.".format(
            training_set.finished_epochs, len(train_pred)))

        if not len(train_pred):
            break
            
        train_acc_epoch = accuracy(np.stack(train_pred), np.stack(train_labels))

        train_log_loss = evaluate_log_loss(train_pred, train_labels)
    
        print('Train log loss error {}.'.format(train_log_loss))
        print('Train set accuracy {}.'.format(train_acc_epoch))
        print('Train set confusion matrix.')
        confusion_matrix = display_confusion_matrix_info(train_labels, train_pred)
        train_sensitivity = get_sensitivity(confusion_matrix)
        train_specificity = get_specificity(confusion_matrix)
        print('Test data sensitivity {} and specificity {}'.format(
            train_sensitivity, train_specificity))

        export_evaluation_summary(train_log_loss, 
                                  train_acc_epoch, 
                                  train_sensitivity, 
                                  step, sess, train_writer)

        print('Evaluate validation set')
        validation_acc, validation_log_loss, val_sensitivity, val_specificity = evaluate_validation_set(sess, 
                                                                                                        validation_set,
                                                                                                        softmax_prediction,
                                                                                                        x,
                                                                                                        batch_size)
        if not validation_log_loss:
            break
        export_evaluation_summary(validation_log_loss, 
                                  validation_acc, 
                                  val_sensitivity, 
                                  step, sess, validation_writer)

        print('Validation accuracy: %.1f%%' % validation_acc)
        print('Log loss overall validation samples: {}.'.format(
            validation_log_loss))
        print('Validation set sensitivity {} and specificity {}'.format(
            val_sensitivity, val_specificity))

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
                               validaton_log_loss_incr_threshold):
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
                      softmax_prediction,
                      x)
