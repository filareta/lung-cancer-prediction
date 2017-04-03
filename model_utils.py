import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


def store_error_plots(validation_err, train_err):
    try:
        plt.plot(validation_err)
        plt.savefig("validation_errors.png")

        plt.plot(train_err)
        plt.savefig("train_errors.png")
    except Exception as e:
        print("Drawing errors failed with: {}".format(e))


# TODO: Think of a better evaluation strategy
def high_error_increase(errors, current, 
    least_count=3, incr_threshold=0.1):
    if len(errors) < least_count:
        return False

    return any(current - x >= incr_threshold 
        for x in errors)


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
