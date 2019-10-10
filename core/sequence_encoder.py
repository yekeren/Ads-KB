from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
from core import utils

from protos import sequence_encoder_pb2


class SequenceEncoder(abc.ABC):
  """Text encoder interface."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    self._model_proto = model_proto
    self._is_training = is_training

  @abc.abstractmethod
  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    pass


class AvgPoolingEncoder(SequenceEncoder):

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    super(AvgPoolingEncoder, self).__init__(model_proto, is_training)

  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    with tf.name_scope('avg_pooling_encoder'):
      mask = tf.sequence_mask(
          length, maxlen=utils.get_tensor_shape(feature)[-2], dtype=tf.float32)
      feature = utils.masked_avg_nd(data=feature, mask=mask, dim=1)
      return tf.squeeze(feature, axis=1)


class AttnPoolingEncoder(SequenceEncoder):

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    super(AttnPoolingEncoder, self).__init__(model_proto, is_training)

  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    mask = tf.sequence_mask(
        length, maxlen=utils.get_tensor_shape(feature)[1], dtype=tf.float32)

    # Compute attention distribution.

    node = feature
    for i in range(options.hidden_layers):
      node = tf.contrib.layers.fully_connected(
          inputs=node,
          num_outputs=feature.get_shape()[-1].value,
          scope=scope + '/hidden_{}'.format(i))
    logits = tf.contrib.layers.fully_connected(
        inputs=node, num_outputs=1, activation_fn=None, scope=scope)

    probas = utils.masked_softmax(
        data=logits, mask=tf.expand_dims(mask, axis=-1), dim=1)
    feature = utils.masked_sum_nd(data=feature * probas, mask=mask, dim=1)

    # Summary.

    #tf.summary.histogram('attn/probas/' + scope, probas)
    #tf.summary.histogram('attn/logits/' + scope, logits)

    return tf.squeeze(feature, axis=1)


class MLPEncoder(SequenceEncoder):

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    super(MLPEncoder, self).__init__(model_proto, is_training)

  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    mask = tf.sequence_mask(
        length, maxlen=utils.get_tensor_shape(feature)[1], dtype=tf.float32)

    feature = tf.contrib.layers.fully_connected(
        inputs=feature,
        num_outputs=feature.get_shape()[-1].value,
        activation_fn=None,
        scope=scope)

    feature = utils.masked_avg_nd(data=feature, mask=mask, dim=1)
    return tf.squeeze(feature, axis=1)


class LSTMEncoder(SequenceEncoder):

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    super(LSTMEncoder, self).__init__(model_proto, is_training)

  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    def lstm_cell():
      cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units=options.hidden_units, forget_bias=1.0)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=options.input_keep_prob,
            output_keep_prob=options.output_keep_prob,
            state_keep_prob=options.state_keep_prob)
      return cell

    rnn_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(options.number_of_layers)])

    with tf.variable_scope(scope):
      outputs, state = tf.nn.dynamic_rnn(
          cell=rnn_cell,
          inputs=feature,
          sequence_length=length,
          parallel_iterations=options.parallel_iterations,
          dtype=tf.float32)

    return state[0].h

    #mask = tf.sequence_mask(
    #    length, maxlen=utils.get_tensor_shape(feature)[1], dtype=tf.float32)

    #feature = utils.masked_avg_nd(data=outputs, mask=mask, dim=1)
    #return tf.squeeze(feature, axis=1)


class BiLSTMEncoder(SequenceEncoder):

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: the actual model proto.
      is_training: if True, training graph will be built.
    """
    super(BiLSTMEncoder, self).__init__(model_proto, is_training)

  def encode(self, feature, length, scope=None):
    """Encodes sequence features into representation.

    Args:
      feature: A [batch, max_sequence_len, dims] float tensor.
      length: A [batch] int tensor.

    Returns:
      A [batch, dims] float tensor.
    """
    options = self._model_proto
    is_training = self._is_training

    def lstm_cell():
      cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units=options.hidden_units, forget_bias=1.0)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=options.input_keep_prob,
            output_keep_prob=options.output_keep_prob,
            state_keep_prob=options.state_keep_prob)
      return cell

    rnn_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(options.number_of_layers)])

    with tf.variable_scope(scope):
      outputs, state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=rnn_cell,
          cell_bw=rnn_cell,
          inputs=feature,
          sequence_length=length,
          parallel_iterations=options.parallel_iterations,
          dtype=tf.float32)

      mask = tf.sequence_mask(
          length, maxlen=utils.get_tensor_shape(feature)[1], dtype=tf.float32)

      # outputs = tf.multiply(0.5, outputs[0] + outputs[1])
      # feature = utils.masked_avg_nd(data=outputs, mask=mask, dim=1)
      # return tf.squeeze(feature, axis=1)

      state_list = []
      for state_per_direction in state:
        for state_per_layer in state_per_direction:
          state_list.extend([state_per_layer.c, state_per_layer.h])

      state_final = tf.contrib.layers.fully_connected(
          inputs=tf.concat(state_list, axis=-1),
          num_outputs=options.output_units,
          activation_fn=None,
          scope='bilstm_output')

    return state_final


def build(options, is_training=False):
  """Builds sequence encoder.

  Args:
    options: An instance of SequenceEncoder proto.
    is_training: If true, build the training graph.
  """
  if not isinstance(options, sequence_encoder_pb2.SequenceEncoder):
    raise ValueError(
        'The options has to be an instance of sequence_encoder_pb2.SequenceEncoder.'
    )

  sequence_encoder_oneof = options.WhichOneof('sequence_encoder_oneof')

  if 'lstm_encoder' == sequence_encoder_oneof:
    encoder = LSTMEncoder(options.lstm_encoder, is_training=is_training)
    return encoder

  if 'bilstm_encoder' == sequence_encoder_oneof:
    encoder = BiLSTMEncoder(options.bilstm_encoder, is_training=is_training)
    return encoder

  if 'avg_pooling_encoder' == sequence_encoder_oneof:
    encoder = AvgPoolingEncoder(
        options.avg_pooling_encoder, is_training=is_training)
    return encoder

  if 'attn_pooling_encoder' == sequence_encoder_oneof:
    encoder = AttnPoolingEncoder(
        options.pooling_encoder, is_training=is_training)
    return encoder

  if 'mlp_encoder' == sequence_encoder_oneof:
    encoder = MLPEncoder(options.pooling_encoder, is_training=is_training)
    return encoder

  raise ValueError('Invalid sequence encoder {}'.format(sequence_encoder_oneof))
