from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from ..base_tf_model import BaseTFModel
from ...util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ...util.pooling import mean_pool
from ...util.rnn import last_relevant_output

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
        self.att_dim =  2*self.rnn_hidden_size #config_dict.pop("att_dim")

        self.multi_ATT1 = tf.get_variable(name = 'a1', shape = (self.att_dim, 2*self.rnn_hidden_size ))
        self.multi_ATT2 = tf.get_variable(name = 'a2', shape = (2*self.rnn_hidden_size, self.att_dim))

        self.ATT1 = tf.get_variable(name = 'a3', shape = (self.att_dim, 2*self.rnn_hidden_size ))
        self.ATT2 = tf.get_variable(name = 'a4', shape = (1, self.att_dim))

        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

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

    @overrides
    def _build_forward(self):
        """
        Using the values in the config passed to the SiameseBiLSTM object
        on creation, build the forward pass of the computation graph.
        """
        # A mask over the word indices in the sentence, indicating
        # which indices are padding and which are words.
        # Shape: (batch_size, num_sentence_words)
        sentence_one_mask = tf.sign(self.sentence_one,
                                    name="sentence_one_masking")
        sentence_two_mask = tf.sign(self.sentence_two,
                                    name="sentence_two_masking")

        # The unpadded lengths of sentence one and sentence two
        # Shape: (batch_size,)
        sentence_one_len = tf.reduce_sum(sentence_one_mask, 1)
        sentence_two_len = tf.reduce_sum(sentence_two_mask, 1)

        word_vocab_size = self.word_vocab_size
        word_embedding_dim = self.word_embedding_dim
        word_embedding_matrix = self.word_embedding_matrix
        fine_tune_embeddings = self.fine_tune_embeddings

        with tf.variable_scope("embeddings"):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self.mode == "train":
                    # Load the word embedding matrix that was passed in
                    # since we are training
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[word_vocab_size,
                               word_embedding_dim],
                        initializer=tf.constant_initializer(
                            word_embedding_matrix),
                        trainable=fine_tune_embeddings)
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrix already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[word_vocab_size,
                                                          word_embedding_dim],
                                                   dtype="float",
                                                   trainable=fine_tune_embeddings)

            with tf.variable_scope("word_embeddings"):
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_one = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_one)
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two)

        rnn_hidden_size = self.rnn_hidden_size
        rnn_output_mode = self.rnn_output_mode
        output_keep_prob = self.output_keep_prob
        rnn_cell_fw_one = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell_fw_one = SwitchableDropoutWrapper(rnn_cell_fw_one,
                                                     self.is_train,
                                                     output_keep_prob=output_keep_prob)
        rnn_cell_bw_one = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell_bw_one = SwitchableDropoutWrapper(rnn_cell_bw_one,
                                                     self.is_train,
                                                     output_keep_prob=output_keep_prob)
        with tf.variable_scope("encode_sentences"):
            # Encode the first sentence.
            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype="float",
                sequence_length=sentence_one_len,
                inputs=word_embedded_sentence_one,
                scope="encoded_sentence_one")
            if self.share_encoder_weights:
                # Encode the second sentence, using the same RNN weights.
                tf.get_variable_scope().reuse_variables()
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell_fw_one,
                    cell_bw=d_rnn_cell_bw_one,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_one")
            else:
                # Encode the second sentence with a different RNN
                rnn_cell_fw_two = LSTMCell(rnn_hidden_size, state_is_tuple=True)
                d_rnn_cell_fw_two = SwitchableDropoutWrapper(
                    rnn_cell_fw_two,
                    self.is_train,
                    output_keep_prob=output_keep_prob)
                rnn_cell_bw_two = LSTMCell(rnn_hidden_size, state_is_tuple=True)
                d_rnn_cell_bw_two = SwitchableDropoutWrapper(
                    rnn_cell_bw_two,
                    self.is_train,
                    output_keep_prob=output_keep_prob)
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell_fw_two,
                    cell_bw=d_rnn_cell_bw_two,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_two")

            # Now, combine the fw_output and bw_output for the
            # first and second sentence LSTM outputs
            if rnn_output_mode == "last":
                # Get the last unmasked output from the RNN
                last_fw_output_one = last_relevant_output(fw_output_one,
                                                          sentence_one_len)
                last_bw_output_one = last_relevant_output(bw_output_one,
                                                          sentence_one_len)
                last_fw_output_two = last_relevant_output(fw_output_two,
                                                          sentence_two_len)
                last_bw_output_two = last_relevant_output(bw_output_two,
                                                          sentence_two_len)
                # Shape: (batch_size, 2*rnn_hidden_size)
                encoded_sentence_one = tf.concat([last_fw_output_one,
                                                  last_bw_output_one], 1)
                encoded_sentence_two = tf.concat([last_fw_output_two,
                                                  last_bw_output_two], 1)
            elif rnn_output_mode == 'H':
            	#fw_output_one  batch_size X T X d 
            	#bw_output_one
            	#fw_output_two
            	#bw_output_two

            	# Shape: (batch_size, 2*rnn_hidden_size)

                # ADD POSITIONAL ENCODING?

                H1 = tf.concat([fw_output_one, bw_output_one], 1)
                H2 = tf.concat([fw_output_two, bw_output_two], 1)

                A1 = self.multi_attention(H1)
                A2 = self.multi_attention(H2)

                M1 = tf.einsum('ijk,ij->ik', A1, H1)
                M2 = tf.einsum('ijk,ij->ik', A2, H2)

                # ADD POSITIONAL ENCODING?

                a1 = self.reg_attention(M1)
                a2 = self.reg_attention(M2)

                encoded_sentence_one = tf.einsum('ijk,ij->ik', a1, M1)
                encoded_sentence_two = tf.einsum('ijk,ij->ik', a2, M2)



            else:
                raise ValueError("Got an unexpected value {} for "
                                 "rnn_output_mode, expected one of "
                                 "[mean_pool, last]")

        with tf.name_scope("loss"):
            # Use the exponential of the negative L1 distance
            # between the two encoded sentences to get an output
            # distribution over labels.
            # Shape: (batch_size, 2)
            self.y_pred = self._l1_similarity(encoded_sentence_one,
                                              encoded_sentence_two)
            # Manually calculating cross-entropy, since we output
            # probabilities and can't use softmax_cross_entropy_with_logits
            # Add epsilon to the probabilities in order to prevent log(0)
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(tf.cast(self.y_true, "float") *
                               tf.log(self.y_pred),
                               axis=1))

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

    def multi_attention(H):
        B = tf.matmul(self.multi_ATT1, H, transpose_b = True)
        B = tf.tanh(B)
        B = tf.matmul(self.multi_ATT2, B)
        A = tf.nn.softmax(B, axis = -1)
        return A

    def reg_attention(H):
        B = tf.matmul(self.ATT1, H, transpose_b = True)
        B = tf.tanh(B)
        B = tf.matmul(self.ATT2, B)
        A = tf.nn.softmax(B, axis = -1)
        return A

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

    def _triplet_similarity(self, anchor_sentence, positive_sentence, negative_sentence, alpha=0.4):
        """
        Given a triplet of encoded sentences (vectors), return the contrastive loss computed
        over the three inputs.

        Parameters
        ----------
        anchor_sentence: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded anchor sentences to use in the loss calculation.

        positive_sentence: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded positive sentences to use in the loss calculation.

        negative_sentence: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded negative sentences to use in the loss calculation.


        Returns
        -------
        loss: float
            A real number representing the contrastive loss
        """
        positive_distance = tf.reduce_sum(tf.square(anchor_sentence - positive_sentence), axis=1)
        negative_distance = tf.reduce_sum(tf.square(anchor_sentence - negative_sentence), axis=1)
        loss = tf.maximum(positive_distance - negative_distance + alpha)
        return loss
        