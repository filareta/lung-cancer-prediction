import tensorflow as tf
import pandas as pd

import data_set as ds
from model_definition import image_tensor_shape
from model_utils import evaluate_test_set


data_loader = ds.DataLoader()
exact_tests = data_loader.get_exact_tests_set()
test_prediction, tf_test_dataset = None, None

sess = tf.Session()
new_saver = tf.train.import_meta_graph('nodules15.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    if v.name == 'test_prediction:0':
        test_prediction = v
    if v.name == 'test_set:0':
        tf_test_dataset = v


evaluate_test_set(sess, 
                  exact_tests, 
                  image_tensor_shape, 
                  test_prediction, 
                  tf_test_dataset)