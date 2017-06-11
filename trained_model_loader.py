import tensorflow as tf

from utils import store_to_csv
import config

from model_utils import tf_test_dataset, evaluate_test_set
from model_factory import ModelFactory


# Construct model
factory = ModelFactory()
model = factory.get_network_model()

test_prediction = tf.nn.softmax(model.conv_net(tf_test_dataset, 1.0), 
    name='test_prediction')

data_loader = factory.get_data_loader()
test_set = data_loader.get_exact_tests_set()
out_dir = data_loader.results_out_dir()
print(out_dir)


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, out_dir + config.RESTORE_MODEL_CKPT)

    evaluate_test_set(sess, 
                      test_set,
                      test_prediction,
                      tf_test_dataset)