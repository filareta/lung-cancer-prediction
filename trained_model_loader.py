import tensorflow as tf

import data_set as ds
from utils import store_to_csv
import config
from model_definition import x, y, keep_prob, learning_rate,
from model_definition import weights, biases, tf_test_dataset
from model_utils import evaluate_test_sets
from model import conv_net


test_prediction = tf.nn.softmax(conv_net(tf_test_dataset, weights, biases, 1.0), 
    name='test_prediction')

data_loader = ds.DataLoader()
test_set = data_loader.get_exact_tests_set()
test_prediction, tf_test_dataset = None, None
out_dir = data_loader.results_out_dir()
print(out_dir)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, out_dir + config.RESTORE_MODEL_CKPT)

evaluate_test_set(sess, 
                  test_set,
                  test_prediction,
                  tf_test_dataset)