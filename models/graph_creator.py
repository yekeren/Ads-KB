from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from protos import graph_creator_pb2
from core import utils

slim = tf.contrib.slim

_EPSILON = 1e-8


def _concat_batch_2d_tensors(matrice_list):
  """Concatenates A 2-D list of tensors to a single tensor.

  Args:
    matrice_list: A 2D list of [batch, height_{i,j}, width_{i,j}] float tensors.

  Returns:
    A [batch, height, width] single float tensor.
  """
  matrice_list = [tf.concat(x, axis=-1) for x in matrice_list]
  matrix = tf.concat(matrice_list, axis=1)
  return matrix


def _calc_graph_edge_weights(node_to,
                             node_from,
                             context=None,
                             hidden_layers=None,
                             hidden_units=None,
                             dropout_keep_prob=1.0,
                             is_training=False,
                             scope='calc_graph_edge_weight'):
  """Calculates the weights from node_from to node_to.

  Args:
    node_to: A [batch, max_num_node_to, dims] float tensor.
    node_from: A [batch, max_num_node_from, dims] float tensor.
    context: A [batch, dims] float tensor.
    hidden_layers: An integer denoting number of MLP layers.
    hidden_units: An integer denoting MLP hidden units.
    dropout_keep_prob: Keep probability of the dropout layers.
    is_training: If true, build the training graph.
    scope: Variable scope name.

  Returns:
    A [batch, max_num_node_to, max_num_node_from] float tensor.
      The [?, i, j]-th element denotes the edge weight from j to i.
  """
  with tf.variable_scope(scope):
    batch, max_num_node_to, _ = utils.get_tensor_shape(node_to)
    batch, max_num_node_from, _ = utils.get_tensor_shape(node_from)

    # Concatenate the edge features.
    #   inputs = [node_to, node_from, node_to * node_from, context].

    expanded_node_to = tf.tile(
        tf.expand_dims(node_to, axis=2), [1, 1, max_num_node_from, 1])
    expanded_node_from = tf.tile(
        tf.expand_dims(node_from, axis=1), [1, max_num_node_to, 1, 1])

    inputs = [
        expanded_node_to, expanded_node_from,
        tf.multiply(expanded_node_to, expanded_node_from)
    ]
    if context is not None:

      context = tf.expand_dims(tf.expand_dims(context, 1), 1)
      context = tf.tile(context, [1, max_num_node_to, max_num_node_from, 1])
      inputs.extend(
          [context * expanded_node_to, context * expanded_node_from, context])
    inputs = tf.concat(inputs, axis=-1)

    # Multi-layer perceptron.

    hiddens = inputs
    for layer_i in range(hidden_layers):
      hiddens = tf.contrib.layers.fully_connected(
          inputs=hiddens,
          num_outputs=hidden_units,
          activation_fn=tf.nn.tanh,
          scope='hidden_{}'.format(layer_i))
      hiddens = slim.dropout(
          hiddens, dropout_keep_prob, is_training=is_training)

    outputs = tf.contrib.layers.fully_connected(
        inputs=hiddens, num_outputs=1, activation_fn=None, scope='output')
    outputs = tf.squeeze(outputs, axis=-1)

  return outputs


def _calc_graph_node_scores(node,
                            hidden_layers=None,
                            hidden_units=None,
                            dropout_keep_prob=1.0,
                            is_training=False,
                            scope='calc_graph_node_scores'):
  """Calculates the node scores [from node to node].

  Args:
    node: A [batch, max_num_node, dims] float tensor.
    hidden_layers: An integer denoting number of MLP layers.
    hidden_units: An integer denoting MLP hidden units.
    dropout_keep_prob: Keep probability of the dropout layers.
    is_training: If true, build the training graph.
    scope: Variable scope name.

  Returns:
    A [batch, max_num_node] float tensor.
  """
  with tf.variable_scope(scope):
    batch = utils.get_tensor_shape(node)[0]

    # Concatenate the node features, inputs = [node].

    hiddens = node
    for layer_i in range(hidden_layers):
      hiddens = tf.contrib.layers.fully_connected(
          inputs=hiddens,
          num_outputs=hidden_units,
          activation_fn=tf.nn.relu,
          scope='hidden_{}'.format(layer_i))
      hiddens = slim.dropout(
          hiddens, dropout_keep_prob, is_training=is_training)

    outputs = tf.contrib.layers.fully_connected(
        inputs=hiddens, num_outputs=1, activation_fn=None, scope='output')
    outputs = tf.squeeze(outputs, axis=-1)
  return outputs


class GraphCreator(abc.ABC):
  """Graph creator."""

  def __init__(self, options, is_training):
    self._options = options
    self._is_training = is_training

  @abc.abstractmethod
  def create_graph(self, proposal_repr, slogan_repr, dbpedia_repr,
                   proposal_mask, slogan_mask, dbpedia_mask,
                   dbpedia_to_slogan_mask):
    pass


class ConvGraphCreator(GraphCreator):
  """ConvGraphCreator."""

  def __init__(self, options, is_training):
    """Initializes the graph creator."""
    super(ConvGraphCreator, self).__init__(options, is_training)

  def _create_edge_weights_helper(self, node_to, node_from, scope,
                                  context=None):
    """Helper function for creating edges."""
    options = self._options
    is_training = self._is_training

    return _calc_graph_edge_weights(
        node_to,
        node_from,
        context=context,
        hidden_layers=options.edge_mlp_options.hidden_layers,
        hidden_units=options.edge_mlp_options.hidden_units,
        dropout_keep_prob=options.edge_mlp_options.dropout_keep_prob,
        is_training=is_training,
        scope=scope)

  def _create_access_matrix(self,
                            max_proposal_num,
                            max_slogan_num,
                            max_dbpedia_num,
                            dbpedia_to_slogan_mask,
                            batch=1):
    """Creates accessibility matrix. In which, ONE means connected.

    Args:
      max_proposal_num: A scalar tensor denoting maximum number of proposals.
      max_slogan_num: A scalar tensor denoting maximum number of slogans.
      max_dbpedia_num: A scalar tensor denoting maximum number of dbpedia comments.

    Returns:
      A [batch, max_node_num, max_node_num] float tensor.
    """
    options = self._options
    is_training = self._is_training

    with tf.name_scope('create_access_matrix'):

      # Check if proposal feature is enabled.

      if options.feature_indicator in [
          graph_creator_pb2.ConvGraphCreator.ONLY_PROPOSAL,
          graph_creator_pb2.ConvGraphCreator.PROPOSAL_AND_SLOGAN,
          graph_creator_pb2.ConvGraphCreator.ALL,
      ]:
        proposal_to_sentinel_mask = tf.ones([batch, 1, max_proposal_num])
      else:
        proposal_to_sentinel_mask = tf.zeros([batch, 1, max_proposal_num])

      # Check if slogan feature is enabled.

      if options.feature_indicator in [
          graph_creator_pb2.ConvGraphCreator.ONLY_SLOGAN,
          graph_creator_pb2.ConvGraphCreator.PROPOSAL_AND_SLOGAN,
          graph_creator_pb2.ConvGraphCreator.ALL,
      ]:
        slogan_to_sentinel_mask = tf.ones([batch, 1, max_slogan_num])
      else:
        slogan_to_sentinel_mask = tf.zeros([batch, 1, max_slogan_num])

      # Check if dbpedia feature is enabled.

      if options.feature_indicator in [
          graph_creator_pb2.ConvGraphCreator.ALL,
      ]:
        dbpedia_to_slogan_mask = dbpedia_to_slogan_mask
      else:
        dbpedia_to_slogan_mask = tf.zeros(
            [batch, max_slogan_num, max_dbpedia_num])

      # Check if proposal and slogan nodes are connected.

      if options.connect_proposal_and_slogan:
        slogan_to_proposal_mask = tf.ones(
            [batch, max_proposal_num, max_slogan_num])
      else:
        slogan_to_proposal_mask = tf.zeros(
            [batch, max_proposal_num, max_slogan_num])

      def dropout_fn(x, keep_prob=1.0):
        if not is_training:
          return x

        sampled_mask = tf.greater_equal(
            tf.random_uniform(shape=tf.shape(x), minval=0.0, maxval=1.0),
            1 - keep_prob)
        return tf.multiply(x, tf.to_float(sampled_mask))

      access_matrix = []
      access_matrix.append([
          tf.zeros([batch, 1, 1]),
          proposal_to_sentinel_mask,
          slogan_to_sentinel_mask,
          tf.zeros([batch, 1, max_dbpedia_num]),
      ])
      access_matrix.append([
          tf.zeros([batch, max_proposal_num, 1]),
          tf.eye(
              num_rows=max_proposal_num,
              num_columns=max_proposal_num,
              batch_shape=[batch]),
          slogan_to_proposal_mask,
          tf.zeros([batch, max_proposal_num, max_dbpedia_num]),
      ])
      access_matrix.append([
          tf.zeros([batch, max_slogan_num, 1]),
          tf.transpose(slogan_to_proposal_mask, [0, 2, 1]),
          dropout_fn(
              tf.eye(
                  num_rows=max_slogan_num,
                  num_columns=max_slogan_num,
                  batch_shape=[batch]),
              keep_prob=options.graph_slogan_keep_prob),
          dropout_fn(
              dbpedia_to_slogan_mask,
              keep_prob=options.graph_dbpedia_to_slogan_keep_prob),
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

      access_matrix = _concat_batch_2d_tensors(access_matrix)

    return access_matrix

  def _create_lv0_edge_scores(self, proposal_repr, slogan_repr, dbpedia_repr,
                              proposal_mask, slogan_mask, dbpedia_mask):
    """Creates adjacency matrix. Each elem denotes an edge weight.

    Args:
      proposal_repr: A [batch, max_proposal_num, dims] float tensor.
      slogan_repr: A [batch, max_slogan_num, dims] float tensor.

    Returns:
      proposal_scores: A [batch, max_proposal_num] float tensor denoting
        weights of different proposals.
      slogan_scores: A [batch, max_slogan_num] float tensor denoting weights
        of different slogans.
      dbpedia_to_slogan_scores: A [batch, max_slogan_num, max_dbpedia_num] tensor.
    """
    options = self._options
    is_training = self._is_training

    (batch_i, max_proposal_num, max_slogan_num,
     max_dbpedia_num) = (proposal_repr.get_shape()[0].value,
                         utils.get_tensor_shape(proposal_repr)[1],
                         utils.get_tensor_shape(slogan_repr)[1],
                         utils.get_tensor_shape(dbpedia_repr)[1])

    with tf.name_scope('create_lv0_attention_weights'):

      # Process predictions.
      #   slogan_dbpedia_to_proposal_scores shape =
      #     [batch, max_proposal_num, max_slogan_num + max_dbpedia_num].
      #   slogan_dbpedia_to_slogan_scores shape =
      #     [batch, max_slogan_num, max_slogan_num + max_dbpedia_num].

      slogan_dbpedia_repr = tf.concat([slogan_repr, dbpedia_repr], axis=1)

      slogan_dbpedia_to_proposal_scores = self._create_edge_weights_helper(
          proposal_repr,
          slogan_dbpedia_repr,
          scope='slogan_dbpedia_to_proposal_scores')

      slogan_dbpedia_to_slogan_scores = self._create_edge_weights_helper(
          slogan_repr,
          slogan_dbpedia_repr,
          scope='slogan_dbpedia_to_slogan_scores')

      # Compute dbpedia_to_slogan_scores.
      #   slogan_dbpedia_scores shape = [batch, 1, max_slogan_num + max_dbpedia_num]
      #   slogan_dbpedia_to_slogan_scores shape = [batch, max_slogan_num, max_slogan_num + max_dbpedia_num]

      slogan_dbpedia_scores = utils.masked_avg(
          slogan_dbpedia_to_proposal_scores,
          mask=tf.expand_dims(proposal_mask, 2),
          dim=1)

      slogan_dbpedia_to_slogan_scores = tf.add(slogan_dbpedia_scores,
                                               slogan_dbpedia_to_slogan_scores)

      slogan_scores = tf.slice(
          slogan_dbpedia_to_slogan_scores,
          begin=[0, 0, 0],
          size=[batch_i, max_slogan_num, max_slogan_num])
      dbpedia_to_slogan_scores = tf.slice(
          slogan_dbpedia_to_slogan_scores,
          begin=[0, 0, max_slogan_num],
          size=[batch_i, max_slogan_num, max_dbpedia_num])

      slogan_scores = tf.linalg.diag_part(slogan_scores)
      proposal_scores = tf.ones([batch_i, max_proposal_num])

      return proposal_scores, slogan_scores, dbpedia_to_slogan_scores

  def _create_lv1_edge_scores(self, proposal_repr, slogan_repr, proposal_mask,
                              slogan_mask):
    """Creates adjacency matrix. Each elem denotes an edge weight.

    Args:
      proposal_repr: A [batch, max_proposal_num, dims] float tensor.
      slogan_repr: A [batch, max_slogan_num, dims] float tensor.

    Returns:
      proposal_scores: A [batch, max_proposal_num] float tensor denoting
        weights of different proposals.
      slogan_scores: A [batch, max_slogan_num] float tensor denoting weights
        of different slogans.
    """
    options = self._options
    is_training = self._is_training

    with tf.name_scope('create_attention_weights'):

      if options.attention_type == graph_creator_pb2.ConvGraphCreator.CO_ATTENTION:

        # Use co-attention to determine edge importance.
        #   slogan_to_proposal_scores shape = [batch, max_proposal_num, max_slogan_num].
        #   proposal_scores shape = [batch, max_proposal_num]
        #   slogan_scores shape = [batch,  max_slogan_num]
        slogan_to_proposal_scores = self._create_edge_weights_helper(
            proposal_repr, slogan_repr, scope='slogan_to_proposal_scores')
        proposal_scores = utils.masked_avg(
            slogan_to_proposal_scores,
            mask=tf.expand_dims(slogan_mask, 1),
            dim=2)
        proposal_scores = tf.squeeze(proposal_scores, axis=-1)
        slogan_scores = utils.masked_avg(
            slogan_to_proposal_scores,
            mask=tf.expand_dims(proposal_mask, 2),
            dim=1)
        slogan_scores = tf.squeeze(slogan_scores, axis=1)

      elif options.attention_type == graph_creator_pb2.ConvGraphCreator.SELF_ATTENTION:

        # Use self-attention to determine edge importance.
        #   similarity_proposal_proposal shape = [batch, max_proposal_num, max_proposal_num]
        #   similarity_slogan_slogan shape = [batch, max_slogan_num, max_slogan_num]
        #   proposal_scores shape = [batch, 1, max_proposal_num]
        #   slogan_scores shape = [batch, 1, max_slogan_num]

        similarity_proposal_proposal = self._create_edge_weights_helper(
            proposal_repr, proposal_repr, scope='similarity_proposal_proposal')
        similarity_slogan_slogan = self._create_edge_weights_helper(
            slogan_repr, slogan_repr, scope='similarity_slogan_slogan')
        proposal_scores = utils.masked_avg(
            similarity_proposal_proposal,
            tf.expand_dims(proposal_mask, 2),
            dim=1)
        proposal_scores = tf.squeeze(proposal_scores, axis=1)
        slogan_scores = utils.masked_avg(
            similarity_slogan_slogan, tf.expand_dims(slogan_mask, 2), dim=1)
        slogan_scores = tf.squeeze(slogan_scores, axis=1)

      else:
        raise ValueError('Invalid attention type %s' % options.attention_type)

    return proposal_scores, slogan_scores

  def _create_adjacency_matrix(self, proposal_to_sentinel, slogan_to_sentinel,
                               proposal_to_proposal, slogan_to_slogan,
                               dbpedia_to_slogan):
    """Creates adjacency matrix based on the predicted weights.

    Args:
      proposal_to_sentinel: A [batch, 1, max_proposal_num] float tensor.
      slogan_to_sentinel: A [batch, 1, max_slogan_num] float tensor.
      proposal_to_proposal: A [batch, max_proposal_num, max_proposal_num] 
        float tensor.
      slogan_to_slogan: A [batch, max_slogan_num, max_slogan_num] float tensor.
      dbpedia_to_slogan: A [batch, max_slogan_num, max_dbpedia_num] tensor.

    Returns:
      adjacency matrix of shape [batch, max_node_num, max_node_num].
    """
    (batch_i, max_proposal_num, max_slogan_num,
     max_dbpedia_num) = (proposal_to_sentinel.get_shape()[0].value,
                         utils.get_tensor_shape(proposal_to_sentinel)[2],
                         utils.get_tensor_shape(slogan_to_sentinel)[2],
                         utils.get_tensor_shape(dbpedia_to_slogan)[2])

    with tf.name_scope('create_adjacency_matrix'):
      node_to_node = []
      node_to_node.append([
          tf.zeros([batch_i, 1, 1]),
          proposal_to_sentinel,
          slogan_to_sentinel,
          tf.zeros([batch_i, 1, max_dbpedia_num]),
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_proposal_num, 1]),
          proposal_to_proposal,
          tf.zeros([batch_i, max_proposal_num, max_slogan_num]),
          tf.zeros([batch_i, max_proposal_num, max_dbpedia_num]),
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_slogan_num, 1]),
          tf.zeros([batch_i, max_slogan_num, max_proposal_num]),
          slogan_to_slogan,
          dbpedia_to_slogan,
      ])
      node_to_node.append([
          tf.zeros([batch_i, max_dbpedia_num, 1]),
          tf.zeros([batch_i, max_dbpedia_num, max_proposal_num]),
          tf.zeros([batch_i, max_dbpedia_num, max_slogan_num]),
          tf.zeros([batch_i, max_dbpedia_num, max_dbpedia_num]),
      ])
      node_to_node = _concat_batch_2d_tensors(node_to_node)
    return node_to_node

  def create_graph(self, proposal_repr, slogan_repr, dbpedia_repr,
                   proposal_mask, slogan_mask, dbpedia_mask,
                   dbpedia_to_slogan_mask):
    """Creates graph."""
    options = self._options
    is_training = self._is_training

    (batch_i, max_proposal_num, max_slogan_num,
     max_dbpedia_num) = (proposal_repr.get_shape()[0].value,
                         utils.get_tensor_shape(proposal_repr)[1],
                         utils.get_tensor_shape(slogan_repr)[1],
                         utils.get_tensor_shape(dbpedia_repr)[1])

    # Create access matrix.

    access_matrix = self._create_access_matrix(max_proposal_num, max_slogan_num,
                                               max_dbpedia_num,
                                               dbpedia_to_slogan_mask, batch_i)
    tf.summary.histogram('histogram/access_matrix', access_matrix)

    # Get the graph predictions.

    (proposal_scores, slogan_scores) = self._create_lv1_edge_scores(
        proposal_repr, slogan_repr, proposal_mask, slogan_mask)

    dbpedia_to_slogan_scores = self._create_edge_weights_helper(
        slogan_repr, dbpedia_repr, 'dbpedia_to_slogan')

    # Build adjacency matrix.

    node_to_node = self._create_adjacency_matrix(
        proposal_to_sentinel=tf.expand_dims(proposal_scores, 1),
        slogan_to_sentinel=tf.expand_dims(slogan_scores, 1),
        proposal_to_proposal=tf.linalg.diag(proposal_scores),
        slogan_to_slogan=tf.linalg.diag(slogan_scores),
        dbpedia_to_slogan=dbpedia_to_slogan_scores)

    tf.summary.histogram('histogram/adjacency_logits', node_to_node)

    sentinel_mask = tf.ones([batch_i, 1])
    sentinel_repr = tf.zeros([batch_i, 1, proposal_repr.get_shape()[-1].value])

    node_mask = tf.concat(
        [sentinel_mask, proposal_mask, slogan_mask, dbpedia_mask], axis=1)
    node_repr = tf.concat(
        [sentinel_repr, proposal_repr, slogan_repr, dbpedia_repr], axis=1)

    adjacency = utils.masked_softmax(
        node_to_node,
        mask=access_matrix * tf.expand_dims(node_mask, axis=1),
        dim=-1)
    adjacency = tf.multiply(
        adjacency,
        tf.multiply(tf.expand_dims(node_mask, 1), tf.expand_dims(node_mask, 2)))

    for _ in range(2):
      node_repr = tf.matmul(adjacency, node_repr)

    image_repr = node_repr[:, 0, :]

    return image_repr, adjacency, node_to_node


class HierarchicalGraphCreator(ConvGraphCreator):
  """HierarchicalGraphCreator."""

  def __init__(self, options, is_training):
    """Initializes the graph creator."""
    super(HierarchicalGraphCreator, self).__init__(options, is_training)

  def create_graph(self, proposal_repr, slogan_repr, dbpedia_repr,
                   proposal_mask, slogan_mask, dbpedia_mask,
                   dbpedia_to_slogan_mask):
    """Creates graph."""
    options = self._options
    is_training = self._is_training

    (batch_i, embedding_dims, max_proposal_num, max_slogan_num,
     max_dbpedia_num) = (proposal_repr.get_shape()[0].value,
                         proposal_repr.get_shape()[-1].value,
                         utils.get_tensor_shape(proposal_repr)[1],
                         utils.get_tensor_shape(slogan_repr)[1],
                         utils.get_tensor_shape(dbpedia_repr)[1])

    # Create access matrix.

    access_matrix = self._create_access_matrix(max_proposal_num, max_slogan_num,
                                               max_dbpedia_num,
                                               dbpedia_to_slogan_mask, batch_i)
    tf.summary.histogram('histogram/access_matrix', access_matrix)

    sentinel_mask = tf.ones([batch_i, 1])
    sentinel_repr = tf.zeros([batch_i, 1, proposal_repr.get_shape()[-1].value])
    node_mask = tf.concat(
        [sentinel_mask, proposal_mask, slogan_mask, dbpedia_mask], axis=1)

    # Layer level-0 inference.

    with tf.variable_scope('layer_lv0_inference'):

      # lv0 predictions.

      (lv0_proposal_scores, lv0_slogan_scores,
       lv0_dbpedia_to_slogan_scores) = self._create_lv0_edge_scores(
           proposal_repr, slogan_repr, dbpedia_repr, proposal_mask, slogan_mask,
           dbpedia_mask)

      # Create lv0 graph, edges to sentinel are not updated.

      node_to_node = self._create_adjacency_matrix(
          proposal_to_sentinel=tf.zeros([batch_i, 1, max_proposal_num]),
          slogan_to_sentinel=tf.zeros([batch_i, 1, max_slogan_num]),
          proposal_to_proposal=tf.linalg.diag(lv0_proposal_scores),
          slogan_to_slogan=tf.linalg.diag(lv0_slogan_scores),
          dbpedia_to_slogan=lv0_dbpedia_to_slogan_scores)

      adjacency = utils.masked_softmax(
          node_to_node,
          mask=tf.multiply(access_matrix, tf.expand_dims(node_mask, axis=1)),
          dim=-1)
      adjacency = tf.multiply(
          adjacency,
          tf.multiply(
              tf.expand_dims(node_mask, 1), tf.expand_dims(node_mask, 2)))

      node_repr = tf.concat(
          [sentinel_repr, proposal_repr, slogan_repr, dbpedia_repr], axis=1)
      node_repr = tf.matmul(adjacency, node_repr)

    # Layer level-1 inference.

    with tf.variable_scope('layer_lv1_inference'):

      # Update representation.

      proposal_repr = tf.slice(
          node_repr,
          begin=[0, 1, 0],
          size=[batch_i, max_proposal_num, embedding_dims])
      slogan_repr = tf.slice(
          node_repr,
          begin=[0, 1 + max_proposal_num, 0],
          size=[batch_i, max_slogan_num, embedding_dims])

      # lv1 predictions.

      (lv1_proposal_scores, lv1_slogan_scores) = self._create_lv1_edge_scores(
          proposal_repr, slogan_repr, proposal_mask, slogan_mask)

      # Create lv1 graph, update edges between nodes and the sentinel.

      node_to_node = self._create_adjacency_matrix(
          proposal_to_sentinel=tf.expand_dims(lv1_proposal_scores, 1),
          slogan_to_sentinel=tf.expand_dims(lv1_slogan_scores, 1),
          proposal_to_proposal=tf.linalg.diag(lv0_proposal_scores),
          slogan_to_slogan=tf.linalg.diag(lv0_slogan_scores),
          dbpedia_to_slogan=lv0_dbpedia_to_slogan_scores)

      adjacency = utils.masked_softmax(
          node_to_node,
          mask=access_matrix * tf.expand_dims(node_mask, axis=1),
          dim=-1)
      adjacency = tf.multiply(
          adjacency,
          tf.multiply(
              tf.expand_dims(node_mask, 1), tf.expand_dims(node_mask, 2)))

      node_repr = tf.concat(
          [sentinel_repr, proposal_repr, slogan_repr, dbpedia_repr], axis=1)
      node_repr = tf.matmul(adjacency, node_repr)

    tf.summary.histogram('histogram/adjacency_logits', node_to_node)

    # Sparse loss.
    self_loop_values = tf.linalg.diag_part(adjacency)
    slogan_values = tf.slice(
        self_loop_values,
        begin=[0, 1 + max_proposal_num],
        size=[batch_i, max_slogan_num])
    if options.HasField('sparse_loss_weight'):
      slogan_value_masks = tf.less(slogan_values, 1)
      sparse_loss = -tf.div(
          tf.reduce_sum(tf.boolean_mask(slogan_values, slogan_value_masks)),
          1e-8 + tf.reduce_sum(tf.to_float(slogan_value_masks)))

      tf.summary.scalar('loss/sparse_loss', sparse_loss)
      tf.losses.add_loss(
          tf.multiply(
              sparse_loss, options.sparse_loss_weight, name='sparse_loss'))

    image_repr = node_repr[:, 0, :]
    return image_repr, adjacency, node_to_node, slogan_values


def build_graph_creator(config, is_training):
  """Builds graph creator based on the config."""
  if not isinstance(config, graph_creator_pb2.GraphCreator):
    raise ValueError('Config has to be an instance of GraphCreator proto.')

  graph_creator_oneof = config.WhichOneof('graph_creator_oneof')

  if 'conv_graph_creator' == graph_creator_oneof:
    return ConvGraphCreator(config.conv_graph_creator, is_training)

  if 'hierarchical_graph_creator' == graph_creator_oneof:
    return HierarchicalGraphCreator(config.hierarchical_graph_creator,
                                    is_training)

  raise ValueError('Invalid graph creator %s' % graph_creator_oneof)
