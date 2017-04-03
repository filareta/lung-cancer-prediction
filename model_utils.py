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