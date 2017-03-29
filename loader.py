import tensorflow as tf
import pandas as pd

import compress_dicoms as cd
import input_dicoms as ind
import data_set as ds


num_channels = 1
n_x = ind.IMAGE_PXL_SIZE
n_y = ind.IMAGE_PXL_SIZE
n_z = ind.HM_SLICES

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
        #[first class=no cancer=0, second class=cancer=1]
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