from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from models.model_base import ModelBase
from protos import match_gcn_model_pb2

from core import utils
from core.training_utils import build_hyperparams
from models import utils as model_utils
from reader.advise_reader import InputDataFields
from core import sequence_encoder

from models.registry import register_model_class

slim = tf.contrib.slim

_FIELD_IMAGE_ID = 'image_id'
_FIELD_IMAGE_IDS_GATHERED = 'image_ids_gathered'
_FIELD_SIMILARITY = 'similarity'
_FIELD_ADJACENCY = 'adjacency'
_FIELD_ADJACENCY_LOGITS = 'adjacency_logits'


class Model(ModelBase):
  """ADVISE model."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: an instance of match_gcn_model_pb2.AdViSEGCN
      is_training: if True, training graph will be built.
    """
    super(Model, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, match_gcn_model_pb2.MatchGCNModel):
      raise ValueError('The model_proto has to be an instance of AdViSEGCN.')

    options = model_proto

    # Read vocabulary.

    def filter_fn(word_with_freq, min_freq):
      return [word for word, freq in word_with_freq if freq > min_freq]

    stmt_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.stmt_vocab_list_path)
    stmt_vocab_list = filter_fn(stmt_vocab_with_freq, 5)

    slgn_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.slgn_vocab_list_path)
    slgn_vocab_list = filter_fn(slgn_vocab_with_freq, 20)

    slgn_dbpedia_vocab_with_freq = model_utils.read_vocabulary_with_frequency(
        options.slgn_kb_vocab_list_path)
    slgn_dbpedia_vocab_list = filter_fn(slgn_dbpedia_vocab_with_freq, 20)

    vocab_list = sorted(
        set(stmt_vocab_list + slgn_vocab_list + slgn_dbpedia_vocab_list))
    tf.logging.info('Vocab, len=%i', len(vocab_list))

    # Read glove data.

    (word2vec_dict,
     embedding_dims) = model_utils.load_glove_data(options.glove_path)

    oov_word, glove_word, glove_vec = [], [], []

    for word in vocab_list:
      if not word in word2vec_dict:
        oov_word.append(word)
      else:
        glove_word.append(word)
        glove_vec.append(word2vec_dict[word])

    self._embedding_dims = embedding_dims
    self._shared_vocab = glove_word + oov_word + ['out-of-vocabulary']
    self._glove_vec = np.stack(glove_vec, 0)

    tf.logging.info('Vocab, glove=%i, all=%i', len(glove_word),
                    len(self._shared_vocab))

    # Text encoder.

    self._text_encoder = sequence_encoder.build(
        options.text_encoder, is_training=is_training)

  def _mask_groundtruth(self, groundtruth_strings, question_strings):
    """Gets groundtruth mask from groundtruth_strings and question_strings.

    Args:
      groundtruth_strings: A [batch_groundtruth, max_groundtruth_text_len] string tensor.
      question_strings: A [batch_question, max_question_text_len] string tensor.

    Returns:
      groundtruth_mask: A [batch_question] boolean tensor, in which `True` 
        denotes the option is correct.
    """
    with tf.name_scope('mask_groundtruth_op'):
      groundtruth_strings = tf.string_strip(
          tf.reduce_join(groundtruth_strings, axis=-1, separator=' '))
      question_strings = tf.string_strip(
          tf.reduce_join(question_strings, axis=-1, separator=' '))
      equal_mat = tf.equal(
          tf.expand_dims(question_strings, axis=1),
          tf.expand_dims(groundtruth_strings, axis=0))
      return tf.reduce_any(equal_mat, axis=-1)

  def _create_word_embedding_weights(self,
                                     vocab_list,
                                     glove_vec,
                                     embedding_dims,
                                     trainable,
                                     scope=None):
    """Gets the word embedding.

    Args:
      vocab_list: A list of string denoting the vocabulary_list.
      glove_vec: A [num_words, embedding_dims] numpy array.
      embedding_dims: Dimensions of word embedding vectors.
      scope: Variable scope.

    Returns:
      word_embedding: A tensor of shape [1 + number_of_tokens, embedding_dims].
    """
    with tf.variable_scope(scope):
      glove_embedding = tf.get_variable(
          name='glove_embedding',
          initializer=glove_vec.astype(np.float32),
          trainable=trainable)
      oov_embedding = tf.get_variable(
          name='oov_embedding',
          shape=[len(vocab_list) - len(glove_vec), embedding_dims],
          initializer=tf.initializers.random_uniform(-0.01, 0.01),
          trainable=True)
      word_embedding = tf.concat([glove_embedding, oov_embedding], 0)
    return word_embedding

  def _node_scores(self,
                   nodes,
                   hidden_layers=None,
                   hidden_units=None,
                   dropout_keep_prob=1.0,
                   is_training=False,
                   scope='node_score'):
    """Connects from N nodes to M nodes.

    Args:
      nodes: A [batch, max_num_nodes, dims] float tensor.
      hidden_layers: An integer denoting number of MLP layers.
      hidden_units: An integer denoting MLP hidden units.
      dropout_keep_prob: Keep probability of the dropout layers.
      is_training: If true, build the training graph.

    Returns:
      A [batch, max_num_nodes1, max_num_nodes2] float tensor.
    """
    with tf.variable_scope(scope):
      batch = utils.get_tensor_shape(nodes)[0]

      hiddens = nodes
      for layer_i in range(hidden_layers):
        hiddens = tf.contrib.layers.fully_connected(
            inputs=hiddens,
            num_outputs=hidden_units,
            scope='hidden_{}'.format(layer_i + 1))
        hiddens = slim.dropout(
            hiddens, dropout_keep_prob, is_training=is_training)

      outputs = tf.contrib.layers.fully_connected(
          inputs=hiddens, num_outputs=1, activation_fn=None, scope='output')
      outputs = tf.squeeze(outputs, axis=-1)
    return outputs

  def _edge_weights_ntom(self,
                         nodes1,
                         nodes2,
                         context=None,
                         hidden_layers=None,
                         hidden_units=None,
                         dropout_keep_prob=1.0,
                         is_training=False,
                         scope='edge_weights_ntom'):
    """Connects from N nodes to M nodes.

    Args:
      nodes1: A [batch, max_num_nodes1, dims] float tensor.
      nodes2: A [batch, max_num_nodes2, dims] float tensor.
      context: A [batch, dims] float tensor.
      hidden_layers: An integer denoting number of MLP layers.
      hidden_units: An integer denoting MLP hidden units.
      dropout_keep_prob: Keep probability of the dropout layers.
      is_training: If true, build the training graph.

    Returns:
      A [batch, max_num_nodes1, max_num_nodes2] float tensor.
    """
    with tf.variable_scope(scope):
      batch = utils.get_tensor_shape(nodes1)[0]
      max_num_nodes1 = utils.get_tensor_shape(nodes1)[1]
      max_num_nodes2 = utils.get_tensor_shape(nodes2)[1]

      expanded_nodes1 = tf.tile(
          tf.expand_dims(nodes1, axis=2), [1, 1, max_num_nodes2, 1])
      expanded_nodes2 = tf.tile(
          tf.expand_dims(nodes2, axis=1), [1, max_num_nodes1, 1, 1])

      inputs = [
          expanded_nodes1, expanded_nodes2,
          tf.multiply(expanded_nodes1, expanded_nodes2)
      ]
      if context is not None:
        context = tf.expand_dims(tf.expand_dims(context, 1), 1)
        context = tf.tile(context, [1, max_num_nodes1, max_num_nodes2, 1])
        inputs.extend(
            [context, context * expanded_nodes1, context * expanded_nodes2])

      inputs = tf.concat(inputs, axis=-1)

      hiddens = inputs
      for layer_i in range(hidden_layers):
        hiddens = tf.contrib.layers.fully_connected(
            inputs=hiddens,
            num_outputs=hidden_units,
            scope='hidden_{}'.format(layer_i + 1))
        hiddens = slim.dropout(
            hiddens, dropout_keep_prob, is_training=is_training)

      outputs = tf.contrib.layers.fully_connected(
          inputs=hiddens, num_outputs=1, activation_fn=None, scope='output')
      outputs = tf.squeeze(outputs, axis=-1)
    return outputs

  def _create_adjacency_matrix(self, sentinel_repr, proposal_repr, slogan_repr,
                               dbpedia_repr, options, is_training):
    """Creates adjacency matrix. Each elem denotes an edge weight.

    Args:
      sentinel_repr: A [batch, 1, dims] float tensor, representation of
        sentinel node.
      proposal_repr: A [batch, max_proposal_num, dims] float tensor.
      slogan_repr: A [batch, max_slogan_num, dims] float tensor.
      dbpedia_repr: A [batch, max_dbpedia_num, dims] float tensor.
    """

    def create_edge_helper(node1, node2, scope, context=None):
      """Helper function for creating edges."""
      return self._edge_weights_ntom(
          node1,
          node2,
          context=context,
          hidden_layers=options.edge_mlp_hidden_layers,
          hidden_units=options.edge_mlp_hidden_units,
          dropout_keep_prob=options.edge_mlp_dropout_keep_prob,
          is_training=is_training,
          scope=scope)

    def node_score_helper(node, scope):
      """Helper function for creating edges."""
      return self._node_scores(
          node,
          hidden_layers=options.edge_mlp_hidden_layers,
          hidden_units=options.edge_mlp_hidden_units,
          dropout_keep_prob=options.edge_mlp_dropout_keep_prob,
          is_training=is_training,
          scope=scope)

    batch_i = sentinel_repr.get_shape()[0].value

    (max_proposal_num, max_slogan_num,
     max_dbpedia_num) = (utils.get_tensor_shape(proposal_repr)[1],
                         utils.get_tensor_shape(slogan_repr)[1],
                         utils.get_tensor_shape(dbpedia_repr)[1])
    node_repr = tf.concat(
        [sentinel_repr, proposal_repr, slogan_repr, dbpedia_repr], axis=1)
    max_node_num = utils.get_tensor_shape(node_repr)[1]

    with tf.name_scope('create_adjacency_matrix'):
      node_to_node = []
      node_to_node.append([
          tf.zeros([batch_i, 1, 1]),
          create_edge_helper(sentinel_repr, proposal_repr,
                             'proposal_to_sentinel'),
          create_edge_helper(sentinel_repr, slogan_repr, 'slogan_to_sentinel'),
          tf.zeros([batch_i, 1, max_dbpedia_num]),
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_proposal_num, 1]),
          tf.zeros([batch_i, max_proposal_num, max_proposal_num]),
          tf.zeros([batch_i, max_proposal_num, max_slogan_num]),
          tf.zeros([batch_i, max_proposal_num, max_dbpedia_num]),
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_slogan_num, 1]),
          tf.zeros([batch_i, max_slogan_num, max_proposal_num]),
          tf.zeros([batch_i, max_slogan_num, max_slogan_num]),
          create_edge_helper(
              slogan_repr,
              dbpedia_repr,
              'dbpedia_to_slogan',
              context=tf.squeeze(sentinel_repr, axis=1)
              if options.use_context else None),
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_dbpedia_num, 1]),
          tf.zeros([batch_i, max_dbpedia_num, max_proposal_num]),
          tf.zeros([batch_i, max_dbpedia_num, max_slogan_num]),
          tf.zeros([batch_i, max_dbpedia_num, max_dbpedia_num]),
      ])
      node_to_node = [tf.concat(x, axis=-1) for x in node_to_node]
      node_to_node = tf.concat(node_to_node, axis=1)

      node_score = node_score_helper(node_repr, scope='node_score')
      node_score = tf.linalg.diag(node_score)

      node_to_node = tf.where(
          tf.eye(
              num_rows=max_node_num,
              num_columns=max_node_num,
              batch_shape=[batch_i],
              dtype=tf.bool),
          x=node_score,
          y=node_to_node)
    return node_to_node

  def _create_access_matrix(self,
                            max_proposal_num,
                            max_slogan_num,
                            max_dbpedia_num,
                            dbpedia_slogan_mask,
                            batch=1,
                            use_dbpedia_knowledge=True):
    """Creates accessibility matrix. In which, ONE means connected.

    Args:
      max_proposal_num: A scalar tensor denoting maximum number of proposals.
      max_slogan_num: A scalar tensor denoting maximum number of slogans.
      max_dbpedia_num: A scalar tensor denoting maximum number of dbpedia comments.

    Returns:
      A [batch, max_node_num, max_node_num] float tensor.
    """
    with tf.name_scope('create_access_matrix'):
      access_matrix = []
      access_matrix.append([
          tf.zeros([batch, 1, 1]),
          tf.ones([batch, 1, max_proposal_num]),
          tf.ones([batch, 1, max_slogan_num]),
          tf.zeros([batch, 1, max_dbpedia_num]),
      ])
      access_matrix.append([
          tf.zeros([batch, max_proposal_num, 1]),
          tf.eye(
              num_rows=max_proposal_num,
              num_columns=max_proposal_num,
              batch_shape=[batch]),
          tf.zeros([batch, max_proposal_num, max_slogan_num]),
          tf.zeros([batch, max_proposal_num, max_dbpedia_num]),
      ])
      access_matrix.append([
          tf.zeros([batch, max_slogan_num, 1]),
          tf.zeros([batch, max_slogan_num, max_proposal_num]),
          tf.eye(
              num_rows=max_slogan_num,
              num_columns=max_slogan_num,
              batch_shape=[batch]),
          dbpedia_slogan_mask if use_dbpedia_knowledge else
          (tf.zeros([batch, max_slogan_num, max_dbpedia_num]))
      ])
      access_matrix.append([
          tf.zeros([batch, max_dbpedia_num, 1]),
          tf.zeros([batch, max_dbpedia_num, max_proposal_num]),
          tf.zeros([batch, max_dbpedia_num, max_slogan_num]),
          tf.eye(
              num_rows=max_dbpedia_num,
              num_columns=max_dbpedia_num,
              batch_shape=[batch]),
      ])
      access_matrix = [tf.concat(x, axis=-1) for x in access_matrix]
      access_matrix = tf.concat(access_matrix, axis=1)
    return access_matrix

  def _text_encoding_helper(self,
                            text_feature,
                            text_length,
                            scope='text_encoding_helper'):
    """Helper function to extract text features.

    Handles the case of non-texts exist.

    Args:
      text_feature: A [batch, max_num_sents, max_sent_len, dims] tensor.
      text_length: A [batch, max_num_sents] int tensor.

    Returns:
      A [batch, max_num_sents, dims] tensor representing sentence encoding.
    """
    (batch, max_num_sents, max_sent_len,
     dims) = utils.get_tensor_shape(text_feature)

    def _encode_text():
      text_repr_reshaped = self._text_encoder.encode(
          tf.reshape(text_feature, [-1, max_sent_len, dims]),
          tf.reshape(text_length, [-1]),
          scope=scope)
      text_repr = tf.reshape(text_repr_reshaped, [batch, max_num_sents, dims])
      return text_repr

    def _encode_text_invalid():
      return tf.fill(dims=[batch, max_num_sents, dims], value=0.0)

    text_repr = tf.cond(
        max_num_sents > 0, true_fn=_encode_text, false_fn=_encode_text_invalid)
    return text_repr

  def build_prediction(self, examples, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    with slim.arg_scope(
        build_hyperparams(self._model_proto.hyperparams, self._is_training)):
      return self._build_prediction(examples)

  def _build_prediction(self, examples, **kwargs):
    """Builds tf graph for prediction.

    Args:
      examples: dict of input tensors keyed by name.
      prediction_task: the specific prediction task.

    Returns:
      predictions: dict of prediction results keyed by name.
    """
    predictions = {}

    options = self._model_proto
    is_training = self._is_training

    # Decode data fields.

    (image_id, image_feature, proposal_feature, proposal_num, slogan_num,
     slogan_text_string, slogan_text_length, dbpedia_ids, dbpedia_num,
     dbpedia_content_string, dbpedia_content_length, dbpedia_slogan_mask,
     groundtruth_num, groundtruth_text_string, groundtruth_text_length) = (
         examples[InputDataFields.image_id],
         examples[InputDataFields.image_feature],
         examples[InputDataFields.proposal_feature],
         examples[InputDataFields.proposal_num],
         examples[InputDataFields.slogan_num],
         examples[InputDataFields.slogan_text_string],
         examples[InputDataFields.slogan_text_length],
         examples[InputDataFields.slogan_kb_ids],
         examples[InputDataFields.slogan_kb_num],
         examples[InputDataFields.slogan_kb_text_string],
         examples[InputDataFields.slogan_kb_text_length],
         examples[InputDataFields.slogan_kb_mask],
         examples[InputDataFields.groundtruth_num],
         examples[InputDataFields.groundtruth_text_string],
         examples[InputDataFields.groundtruth_text_length])

    # Generate in-batch question list.

    batch_i = image_id.get_shape()[0].value

    if is_training:

      # Sample in-batch negatives for the training mode.

      (image_ids_gathered, stmt_text_string,
       stmt_text_length) = model_utils.gather_in_batch_captions(
           image_id, groundtruth_num, groundtruth_text_string,
           groundtruth_text_length)
    else:

      # Use the negatives in the tf.train.Example for the evaluation mode.

      assert batch_i == 1, "We only support `batch_i == 1` for evaluation."

      (question_num, question_text_string,
       question_text_length) = (examples[InputDataFields.question_num],
                                examples[InputDataFields.question_text_string],
                                examples[InputDataFields.question_text_length])

      stmt_text_string = question_text_string[0][:question_num[0], :]
      stmt_text_length = question_text_length[0][:question_num[0]]

      stmt_mask = self._mask_groundtruth(
          groundtruth_strings=groundtruth_text_string[0]
          [:groundtruth_num[0], :],
          question_strings=question_text_string[0][:question_num[0], :])

      image_ids_gathered = tf.where(
          stmt_mask,
          x=tf.fill(tf.shape(stmt_mask), image_id[0]),
          y=tf.fill(tf.shape(stmt_mask), tf.constant(-1, dtype=tf.int64)))

    # Initialize word embedding.

    table = tf.contrib.lookup.index_table_from_tensor(
        self._shared_vocab, num_oov_buckets=1)
    word_embedding = self._create_word_embedding_weights(
        self._shared_vocab,
        self._glove_vec,
        embedding_dims=self._embedding_dims,
        trainable=options.train_word_embedding,
        scope='shared_embedding')

    if options.project_word_embedding:
      if options.project_use_relu:
        word_embedding = slim.dropout(
            tf.nn.relu(word_embedding), 0.5, is_training=is_training)
      word_embedding = tf.contrib.layers.fully_connected(
          inputs=word_embedding,
          num_outputs=self._embedding_dims,
          activation_fn=None,
          scope='project_word_embedding')

    def embedding_helper(text):
      """Helper function for looking up word embedding. """
      return tf.nn.embedding_lookup(
          params=word_embedding, ids=table.lookup(text))

    # Statement representation.
    #   stmt_repr shape = [batch_c, embedding_dims]

    stmt_repr = self._text_encoder.encode(
        embedding_helper(stmt_text_string),
        stmt_text_length,
        scope='statement_encoding'
        if not options.shared_encoding else 'shared_encoding')

    # Slogan representation.
    #   slogan_repr shape = [batch_i, max_slogan_num, embedding_dims]
    #   slogan_mask shape = [batch_i, max_slogan_num]

    with tf.variable_scope(
        tf.get_variable_scope(), reuse=options.shared_encoding):
      slogan_repr = self._text_encoding_helper(
          embedding_helper(slogan_text_string),
          slogan_text_length,
          scope='slogan_encoding'
          if not options.shared_encoding else 'shared_encoding')
    slogan_mask = tf.sequence_mask(
        slogan_num, maxlen=tf.shape(slogan_text_string)[1], dtype=tf.float32)

    # DBPedia representation.
    #   dbpedia_repr shape = [batch_i, max_dbpedia_num, feature_dims]
    #   dbpedia_mask shape = [batch_i, max_dbpedia_num]

    with tf.variable_scope(
        tf.get_variable_scope(), reuse=options.shared_encoding):
      dbpedia_repr = self._text_encoding_helper(
          embedding_helper(dbpedia_content_string),
          dbpedia_content_length,
          scope='knowledge_encoding'
          if not options.shared_encoding else 'shared_encoding')
    dbpedia_mask = tf.sequence_mask(
        dbpedia_num,
        maxlen=tf.shape(dbpedia_content_string)[1],
        dtype=tf.float32)

    # Image representation.
    #   proposal_mask shape = [batch_i, max_proposal_num]
    #   proposal_repr shape = [batch_i, max_proposal_num, feature_dims]

    proposal_mask = tf.sequence_mask(
        proposal_num, maxlen=tf.shape(proposal_feature)[1], dtype=tf.float32)
    proposal_repr = tf.contrib.layers.fully_connected(
        inputs=proposal_feature,
        num_outputs=self._embedding_dims,
        activation_fn=tf.nn.tanh if options.image_use_tanh else None,
        scope='proposal_encoding')
    if options.image_use_tanh:
      proposal_repr = slim.dropout(proposal_repr, 0.5, is_training=is_training)

    # Sentinel representation.

    max_proposal_num = utils.get_tensor_shape(proposal_repr)[1]
    max_slogan_num = utils.get_tensor_shape(slogan_repr)[1]
    max_dbpedia_num = utils.get_tensor_shape(dbpedia_repr)[1]

    with tf.name_scope('sentinel_embedding'):
      sentinel_mask = tf.ones(shape=[batch_i, 1])
      if match_gcn_model_pb2.AVG_ALL == options.sentinel_init_emb:
        sentinel_repr = 0.5 * (
            utils.masked_avg_nd(data=proposal_repr, mask=proposal_mask, dim=1) +
            utils.masked_avg_nd(data=slogan_repr, mask=slogan_mask, dim=1))
      elif match_gcn_model_pb2.ZERO == options.sentinel_init_emb:
        sentinel_repr = tf.zeros(shape=[batch_i, 1, self._embedding_dims])
      else:
        raise ValueError('Invalid sentinel init embedding.')

    ################################################################
    # Build graph.
    ################################################################

    with tf.name_scope('build_graph'):

      # node_mask shape = [batch_i, max_node_num]
      # node_repr shape = [batch_i, max_node_num, dims]

      node_mask = tf.concat(
          [sentinel_mask, proposal_mask, slogan_mask, dbpedia_mask], axis=1)
      node_repr = tf.concat(
          [sentinel_repr, proposal_repr, slogan_repr, dbpedia_repr], axis=1)
      max_node_num = utils.get_tensor_shape(node_mask)[1]

      # Create adjacency matrix.

      node_to_node = self._create_adjacency_matrix(
          sentinel_repr,
          proposal_repr,
          slogan_repr,
          dbpedia_repr,
          self._model_proto,
          is_training=is_training)

      # Create accessibility matrix.

      access_matrix = self._create_access_matrix(
          max_proposal_num,
          max_slogan_num,
          max_dbpedia_num,
          dbpedia_slogan_mask,
          batch_i,
          use_dbpedia_knowledge=options.use_dbpedia_knowledge)

      if options.normalization_method == match_gcn_model_pb2.SOFTMAX:
        adjacency = utils.masked_softmax(
            node_to_node,
            mask=access_matrix * tf.expand_dims(node_mask, axis=1),
            dim=-1)
        adjacency = tf.multiply(
            adjacency,
            tf.multiply(
                tf.expand_dims(node_mask, axis=1),
                tf.expand_dims(node_mask, axis=2)))

      else:
        raise ValueError('Invalid normalization method.')

    # Graph convolutional network.

    tf.summary.scalar('loss/wv_l2',
                      tf.reduce_mean(tf.norm(word_embedding, axis=-1)))
    tf.summary.scalar('loss/gcn_layers', tf.to_float(options.gcn_layers))
    tf.summary.histogram('histogram/proposal_repr', proposal_repr)
    tf.summary.histogram('histogram/slogan_repr', slogan_repr)

    # Sparse loss.

    with tf.name_scope('sparse_loss'):
      summed_identity_edges = tf.reduce_sum(
          tf.linalg.diag_part(adjacency), axis=-1)
      total_identity_edges = 1 + tf.to_float(proposal_num) + tf.to_float(
          slogan_num)
      sparse_loss = -tf.div(
          tf.reduce_sum(summed_identity_edges),
          tf.reduce_sum(total_identity_edges))

    tf.summary.scalar('loss/sparse_loss', sparse_loss)
    if options.graph_sparse_loss_weight > 0:
      tf.losses.add_loss(
          tf.multiply(
              sparse_loss, options.graph_sparse_loss_weight,
              name='sparse_loss'))

    # GCN and image representation.

    if not options.gcn_learn_weights:
      for _ in range(options.gcn_layers):
        node_repr = tf.matmul(adjacency, node_repr)
    else:
      for layer_i in range(options.gcn_layers):
        node_repr = tf.matmul(adjacency, node_repr)
        previous = node_repr
        node_repr = slim.dropout(
            tf.nn.relu(node_repr), 0.5, is_training=is_training)
        node_repr = tf.contrib.layers.fully_connected(
            inputs=node_repr,
            num_outputs=node_repr.get_shape()[-1].value,
            activation_fn=None,
            scope='gcn_{}'.format(layer_i + 1))
        node_repr = node_repr + previous

    image_repr = node_repr[:, 0, :]

    # Similarity computation.

    similarity = tf.matmul(
        tf.nn.l2_normalize(image_repr, axis=-1),
        tf.nn.l2_normalize(stmt_repr, axis=-1),
        transpose_b=True)

    predictions.update({
        _FIELD_IMAGE_ID: image_id,
        _FIELD_IMAGE_IDS_GATHERED: image_ids_gathered,
        _FIELD_SIMILARITY: similarity,
        _FIELD_ADJACENCY: adjacency,
        _FIELD_ADJACENCY_LOGITS: node_to_node,
    })
    return predictions

  def build_loss(self, predictions, **kwargs):
    """Build tf graph to compute loss.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      loss_dict: dict of loss tensors keyed by name.
    """
    options = self._model_proto

    (image_id, image_ids_gathered,
     similarity) = (predictions[_FIELD_IMAGE_ID],
                    predictions[_FIELD_IMAGE_IDS_GATHERED],
                    predictions[_FIELD_SIMILARITY])

    distance = 1.0 - similarity

    pos_mask = tf.cast(
        tf.equal(
            tf.expand_dims(image_id, axis=1),
            tf.expand_dims(image_ids_gathered, axis=0)), tf.float32)
    neg_mask = 1.0 - pos_mask

    if options.triplet_ap_use_avg:
      distance_ap = utils.masked_avg(distance, pos_mask)
    else:
      distance_ap = utils.masked_maximum(distance, pos_mask)

    # negatives_outside: smallest D_an where D_an > D_ap.

    mask = tf.cast(tf.greater(distance, distance_ap), tf.float32)
    mask = mask * neg_mask
    negatives_outside = utils.masked_minimum(distance, mask)

    # negatives_inside: largest D_an.

    negatives_inside = utils.masked_maximum(distance, neg_mask)

    # distance_an: the semihard negatives.

    mask_condition = tf.greater(tf.reduce_sum(mask, axis=1, keepdims=True), 0.0)

    distance_an = tf.where(mask_condition, negatives_outside, negatives_inside)

    # Triplet loss.

    losses = tf.maximum(distance_ap - distance_an + options.triplet_margin, 0)

    return {
        'triplet_loss': tf.reduce_mean(losses),
    }

  def build_evaluation(self, predictions, **kwargs):
    """Build tf graph to evaluate the model.

    Args:
      predictions: dict of prediction results keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    (image_id, image_ids_gathered,
     similarity) = (predictions[_FIELD_IMAGE_ID],
                    predictions[_FIELD_IMAGE_IDS_GATHERED],
                    predictions[_FIELD_SIMILARITY])

    retrieved_index = tf.argmax(similarity, axis=1)
    predicted_alignment = tf.gather(image_ids_gathered,
                                    tf.argmax(similarity, axis=1))

    accuracy, update_op = tf.metrics.accuracy(image_id, predicted_alignment)

    return {'accuracy': (accuracy, update_op)}


register_model_class(match_gcn_model_pb2.MatchGCNModel.ext, Model)
