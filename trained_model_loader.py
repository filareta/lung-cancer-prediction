import tensorflow as tf
import pandas as pd

import data_set as ds
from utils import store_to_csv
import config


data_loader = ds.DataLoader()
test_set = data_loader.get_exact_tests_set()
test_prediction, tf_test_dataset = None, None
out_dir = data_loader.results_out_dir()
print(out_dir)

sess = tf.Session()
new_saver = tf.train.import_meta_graph(out_dir + '/model_15.ckpt.meta')
new_saver.restore(sess, out_dir + '/model_15.ckpt')
all_vars = tf.get_collection('vars')
for v in all_vars:
    if v.name == 'test_prediction:0':
        test_prediction = v
    if v.name == 'test_set:0':
        tf_test_dataset = v

i = 0
patients, probs = [], []

try:
    while i < test_set.num_samples:
        patient, test_img = test_set.next_patient()
        test_img_reshape = tf.reshape(test_img, 
            shape=[-1, config.SLICES, config.IMAGE_PXL_SIZE_X, config.IMAGE_PXL_SIZE_Y, 1])
        test_img = sess.run(test_img_reshape)
        i += 1
        # returns index of column with highest probability
        # [first class=no cancer=0, second class=cancer=1]
        if len(test_img):
            output = sess.run(test_prediction, 
                feed_dict={tf_test_dataset: test_img})
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

store_to_csv(patients, probs, config.SOLUTION_FILE_PATH)