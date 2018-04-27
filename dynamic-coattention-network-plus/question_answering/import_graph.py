import tensorflow as tf
from tensorflow.train import latest_checkpoint

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            latest_ckpt = latest_checkpoint(loc)
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(latest_ckpt + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.M = self.graph.get_tensor_by_name('sentence_one/encode_sentences/M')

    def run(self, question):
        """ Running the activation operation previously imported """
        return self.sess.run(self.M, feed_dict={"sentence_one:0": question, 'is_train:0': False})