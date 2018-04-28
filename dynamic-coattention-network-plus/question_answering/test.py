import tensorflow as tf
import os, sys

print("Tensorflow Version: ", tf.__version__)

loc = '../../paraphrase-id-tensorflow-master/models/baseline_siamese/00/'
#print(os.listdir(loc))

# Create local graph and use it in the session
graph = tf.Graph()
sess = tf.Session(graph = graph)
with graph.as_default():
#if True:
    latest_ckpt = tf.train.latest_checkpoint(loc)
    print(latest_ckpt) # ../../paraphrase-id-tensorflow-master/models/baseline_siamese/00/baseline_siamese-00-10
    # Import saved model from location 'loc' into local graph
    saver = tf.train.import_meta_graph(latest_ckpt + '.meta', clear_devices=True)
    saver.restore(sess, latest_ckpt)

    M = sess.graph.get_operation_by_name('sentence_one/M')