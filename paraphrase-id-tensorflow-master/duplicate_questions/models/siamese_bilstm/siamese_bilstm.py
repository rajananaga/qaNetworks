from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
print("Tensorflow Version ----------> ", tf.__version__)
from tensorflow.contrib.rnn import LSTMCell

from ..base_tf_model import BaseTFModel
from ...util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ...util.pooling import mean_pool
from ...util.rnn import last_relevant_output
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.base_tf_model import BaseTFModel
from util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from util.pooling import mean_pool
from util.rnn import last_relevant_output
'''

logger = logging.getLogger(__name__)


class SiameseBiLSTM(BaseTFModel):
    """
    Parameters
    ----------
    mode: str
        One of [train|predict], to indicate what you want the model to do.
        If you pick "predict", then you must also supply the path to a
        pretrained model and DataIndexer to load to the ``predict`` method.

    word_vocab_size: int
        The number of unique tokens in the dataset, plus the UNK and padding
        tokens. Alternatively, the highest index assigned to any word, +1.
        This is used by the model to figure out the dimensionality of the
        embedding matrix.

    word_embedding_dim: int
        The length of a word embedding. This is used by
        the model to figure out the dimensionality of the embedding matrix.

    word_embedding_matrix: numpy array, optional if predicting
        A numpy array of shape (word_vocab_size, word_emb_dim).
        word_embedding_matrix[index] should represent the word vector for
        that particular word index. This is used to initialize the
        word embedding matrix in the model, and is optional if predicting
        since we assume that the word embeddings variable will be loaded
        with the model.

    fine_tune_embeddings: boolean
        If true, sets the embeddings to be trainable.

    rnn_hidden_size: int
        The output dimension of the RNN encoder. Note that this model uses a
        bidirectional LSTM, so the actual sentence vectors will be
        of length 2*rnn_hidden_size.

    share_encoder_weights: boolean
        Whether to use the same encoder on both input sentnces (thus
        sharing weights), or a different one for each sentence.

    rnn_output_mode: str
        How to calculate the final sentence representation from the RNN
        outputs. mean pool" indicates that the outputs will be averaged (with
        respect to padding), and "last" indicates that the last
        relevant output will be used as the sentence representation.

    output_keep_prob: float
        The probability of keeping an RNN outputs to keep, as opposed
        to dropping it out.
    """

    @overrides
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        super(SiameseBiLSTM, self).__init__(mode=mode)

        self.word_vocab_size = config_dict.pop("word_vocab_size")
        self.word_embedding_dim = config_dict.pop("word_embedding_dim")
        self.word_embedding_matrix = config_dict.pop("word_embedding_matrix", None)
        self.fine_tune_embeddings = config_dict.pop("fine_tune_embeddings")
        self.rnn_hidden_size = config_dict.pop("rnn_hidden_size")
        self.share_encoder_weights = config_dict.pop("share_encoder_weights")
        self.rnn_output_mode = config_dict.pop("rnn_output_mode")
        self.output_keep_prob = config_dict.pop("output_keep_prob")

        self.num_sentence_words = config_dict.pop("num_sentence_words")
        self.att_dim = self.rnn_hidden_size#config_dict.pop("att_dim")

        trainable = self.mode == 'train'

        self.multi_ATT1 = tf.get_variable(name = 'w1', shape = (2*self.rnn_hidden_size, self.att_dim), trainable = trainable)
        self.multi_ATT2 = tf.get_variable(name = 'w2', shape = (self.att_dim, self.num_sentence_words), trainable = trainable)

        self.ATT1 = tf.get_variable(name = 'w3', shape = (2*self.rnn_hidden_size, self.att_dim), trainable = trainable)
        self.ATT2 = tf.get_variable(name = 'w4', shape = (self.att_dim, 1), trainable = trainable)


        if self.mode == "train":
            # Load the word embedding matrix that was passed in
            # since we are training
            self.word_emb_mat = tf.get_variable(
                "word_emb_mat",
                dtype="float",
                shape=[self.word_vocab_size,
                       self.word_embedding_dim],
                initializer=tf.constant_initializer(
                    self.word_embedding_matrix),
                trainable=self.fine_tune_embeddings)
        else:
            # We are not training, so a model should have been
            # loaded with the embedding matrix already there.
            self.word_emb_mat = tf.get_variable("word_emb_mat",
                                           shape=[self.word_vocab_size,
                                                  self.word_embedding_dim],
                                           dtype="float",
                                           trainable=self.fine_tune_embeddings)

        self.rnn_cell_fw = LSTMCell(self.rnn_hidden_size, state_is_tuple=True)
        self.rnn_cell_bw = LSTMCell(self.rnn_hidden_size, state_is_tuple=True)

        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

    @property
    def var_list(self):
        var_list = [self.multi_ATT1, self.multi_ATT2, self.word_emb_mat]
        var_list.extend(self.rnn_cell_fw.variables)
        var_list.extend(self.rnn_cell_bw.variables)
        return var_list

    @overrides
    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence.
        self.sentence_one = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_one")

        # Shape: (batch_size, num_sentence_words)
        # The second input sentence.
        self.sentence_two = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_two")

        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, 2],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    def process_sentence(self, sentence, scope_name = 'scope'):

        sentence_mask = tf.sign(sentence, name="sentence_masking")
        sentence_len = tf.reduce_sum(sentence_mask, 1)
        
        with tf.variable_scope("word_embeddings"):
            # Shape: (batch_size, num_sentence_words, embedding_dim)
            word_embedded_sentence = tf.nn.embedding_lookup(
                self.word_emb_mat,
                sentence)

        # dropout layers
        d_rnn_cell_fw = SwitchableDropoutWrapper(self.rnn_cell_fw,
                                                     self.is_train,
                                                     output_keep_prob=self.output_keep_prob)
        d_rnn_cell_bw = SwitchableDropoutWrapper(self.rnn_cell_bw,
                                                     self.is_train,
                                                     output_keep_prob=self.output_keep_prob)

        #with tf.variable_scope(scope_name):
            #with tf.variable_scope("encode_sentences"):
                # Encode the first sentence.
        (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=d_rnn_cell_fw,
            cell_bw=d_rnn_cell_bw,
            dtype="float",
            sequence_length=sentence_len,
            inputs=word_embedded_sentence,
            scope="encoded_sentence")

        H = tf.concat([fw_output, bw_output], -1)
        # H1.shape = (?, ?, 512)

        A = self.multi_attention(H)
        # (batch, r, T) *  (batch, T, d) = (batch, r, d)

        #(?, r, 512)
        M = tf.matmul(A, H, name = scope_name + '/' + 'M')

        # ADD POSITIONAL ENCODING?

        # a1 = (?, r)
        a = self.reg_attention(M)

        #(?, r)*(?, r, 512)
        encoded_sentence = tf.squeeze(tf.matmul(a, M), axis = 1)
        return M, A, encoded_sentence

    @overrides
    def _build_forward(self):
        """
        Using the values in the config passed to the SiameseBiLSTM object
        on creation, build the forward pass of the computation graph.
        """
        M1, A1, encoded_sentence_one = self.process_sentence(self.sentence_one, scope_name = 'sentence_one')
        M2, A2, encoded_sentence_two = self.process_sentence(self.sentence_two, scope_name = 'sentence_two')

        with tf.name_scope("loss"):
            # Use the exponential of the negative L1 distance
            # between the two encoded sentences to get an output
            # distribution over labels.
            # Shape: (batch_size, 2)
            self.y_pred = self._l1_similarity(encoded_sentence_one, encoded_sentence_two)
            # Manually calculating cross-entropy, since we output
            # probabilities and can't use softmax_cross_entropy_with_logits
            # Add epsilon to the probabilities in order to prevent log(0)
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(tf.cast(self.y_true, "float") *
                               tf.log(self.y_pred),
                               axis=1))

            self.loss = self.loss + self.matrix_penalty(A1) + self.matrix_penalty(A2) #add matrix penalty

        with tf.name_scope("accuracy"):
            # Get the correct predictions.
            # Shape: (batch_size,) of bool
            correct_predictions = tf.equal(
                tf.argmax(self.y_pred, 1),
                tf.argmax(self.y_true, 1))

            # Cast to float, and take the mean to get accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float"))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(self.loss,
                                                  global_step=self.global_step)

        with tf.name_scope("train_summaries"):
            # Add the loss and the accuracy to the tensorboard summary
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def multi_attention(self, H):
        #  (?, T, 512) * (512, 256) = (?, T, 256)
        B = tf.tensordot(H, self.multi_ATT1, [[2],[0]])
        B = tf.tanh(B)
        B = tf.tensordot(B, self.multi_ATT2, [[2],[0]])
        A = tf.nn.softmax(B, dim = 1)
        A = tf.transpose(A, [0,2,1])
        return A

    def reg_attention(self, M):
        #  (?, r, 512) * (512, 256) = (?, r, 256)
        B = tf.tensordot(M, self.ATT1, [[2],[0]])
        B = tf.tanh(B)
        #  (?, r, 256) * (256, 1) = (?, r, 1)
        B = tf.tensordot(B, self.ATT2, [[2],[0]])
        A = tf.nn.softmax(B, dim = 1)
        return tf.transpose(A, [0,2,1])

    def matrix_penalty(self, A):
        AT = tf.transpose(A, [0,2,1])
        I = tf.expand_dims(tf.eye(self.num_sentence_words), axis = 0)
        loss = tf.matmul(A,AT) - I
        loss = tf.pow(loss, 2)
        loss = tf.reduce_sum(loss)
        return loss

    @overrides
    def _get_train_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets[0],
                     self.is_train: True}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets[0],
                     self.is_train: False}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        inputs, _ = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.is_train: False}
        return feed_dict

    def _l1_similarity(self, sentence_one, sentence_two):
        """
        Given a pair of encoded sentences (vectors), return a probability
        distribution on whether they are duplicates are not with:
        exp(-||sentence_one - sentence_two||)

        Parameters
        ----------
        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_ones to use in the probability calculation.

        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_twos to use in the probability calculation.

        Returns
        -------
        class_probabilities: Tensor
            A tensor of shape (batch_size, 2), represnting the probability
            that a pair of sentences are duplicates as
            [is_not_duplicate, is_duplicate].
        """
        with tf.name_scope("l1_similarity"):
            # Take the L1 norm of the two vectors.
            # Shape: (batch_size, 2*rnn_hidden_size)
            l1_distance = tf.abs(sentence_one - sentence_two)

            # Take the sum for each sentence pair
            # Shape: (batch_size, 1)
            summed_l1_distance = tf.reduce_sum(l1_distance, axis=1,
                                               keep_dims=True)

            # Exponentiate the negative summed L1 distance to get the
            # positive-class probability.
            # Shape: (batch_size, 1)
            positive_class_probs = tf.exp(-summed_l1_distance)

            # Get the negative class probabilities by subtracting
            # the positive class probabilities from 1.
            # Shape: (batch_size, 1)
            negative_class_probs = 1 - positive_class_probs

            # Concatenate the positive and negative class probabilities
            # Shape: (batch_size, 2)
            class_probabilities = tf.concat([negative_class_probs,
                                             positive_class_probs], 1)

            # if class_probabilities has 0's, then taking the log of it
            # (e.g. for cross-entropy loss) will cause NaNs. So we add
            # epsilon and renormalize by the sum of the vector.
            safe_class_probabilities = class_probabilities + 1e-08
            safe_class_probabilities /= tf.reduce_sum(safe_class_probabilities,
                                                      axis=1,
                                                      keep_dims=True)
            return safe_class_probabilities