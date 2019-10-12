from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from core import utils
from protos import reader_pb2


class TFExampleDataFields(object):
  """Names of the fields of the tf.train.Example."""
  image_id = "image_id"

  proposal_box = 'proposal/bbox/'
  proposal_ymin = 'proposal/bbox/ymin'
  proposal_xmin = 'proposal/bbox/xmin'
  proposal_ymax = 'proposal/bbox/ymax'
  proposal_xmax = 'proposal/bbox/xmax'
  proposal_text = 'proposal/bbox/text'
  proposal_feature = 'proposal/bbox/feature'

  slogan_box = 'slogan/bbox/'
  slogan_ymin = 'slogan/bbox/ymin'
  slogan_xmin = 'slogan/bbox/xmin'
  slogan_ymax = 'slogan/bbox/ymax'
  slogan_xmax = 'slogan/bbox/xmax'
  slogan_text = 'slogan/bbox/text'

  groundtruth_text = 'groundtruth/text'
  question_text = 'question/text'


class InputDataFields(object):
  """Names of the input tensors."""
  image_id = 'image_id'

  proposal_num = 'proposal_num'
  proposal_box = 'proposal_box'
  proposal_text_string = 'proposal_text_string'
  proposal_text_length = 'proposal_text_length'
  proposal_feature = 'proposal_feature'

  proposal_label_num = 'proposal_label_num'
  proposal_label_text = 'proposal_label_text'
  proposal_label_mask = 'proposal_label_mask'

  slogan_num = 'slogan_num'
  slogan_box = 'slogan_box'
  slogan_text_string = 'slogan_text_string'
  slogan_text_length = 'slogan_text_length'

  slogan_kb_num = 'slogan_kb_num'
  slogan_kb_text_string = 'slogan_kb_text_string'
  slogan_kb_text_length = 'slogan_kb_text_length'
  slogan_kb_ids = 'slogan_kb_ids'
  slogan_kb_mask = 'slogan_kb_mask'

  groundtruth_num = 'groundtruth_num'
  groundtruth_text_string = 'groundtruth_text_string'
  groundtruth_text_length = 'groundtruth_text_length'
  question_num = 'question_num'
  question_text_string = 'question_text_string'
  question_text_length = 'question_text_length'


def _decode_text_string_and_length(raw_text,
                                   max_string_num=None,
                                   max_string_len=None):
  """Decodes a raw_text tensor to get the token and length.

  Args:
    raw_text: A [num_texts] string tensor.
    max_string_num: Maximum number of strings.
    max_string_len: Maximum length of strings.

  Returns:
    num: A [] int tensor denoting the number of strings.
    string: A [num_texts, max_string_len] string tensor representing the tokens.
    length: A [num_texts] int tensor representing the length.
  """
  tokens = tf.sparse_tensor_to_dense(
      tf.strings.split(raw_text, sep=' '), default_value='')

  if max_string_num is None:
    max_string_num = tf.shape(tokens)[0]
  if max_string_len is None:
    max_string_len = tf.shape(tokens)[1]

  tokens = tokens[:max_string_num, :max_string_len]
  length = tf.reduce_sum(tf.cast(tf.not_equal(tokens, ''), tf.int32), axis=-1)

  return tf.shape(tokens)[0], tokens, length


class KnowledgeBase(object):

  def __init__(self, options):

    def create_table_helper(filename):
      """Helper function for creating hash table."""
      return tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              filename,
              key_dtype=tf.string,
              key_index=0,
              value_dtype=tf.string,
              value_index=1,
              delimiter='\t'),
          default_value='')

    self._options = options

    self._query_to_id_table = create_table_helper(options.query_to_id_file)
    self._id_to_comment_table = create_table_helper(options.id_to_comment_file)

  def process_batch_query(self, query):
    """Queries knowledge base to get comments.
    Args:
      query: A [sent_num, query_num] string tensor.

    Returns:
      kb_num: A [] int tensor.
      kb_ids: A [kb_num] string tensor, each element
        denotes an id entry in the knowledge base.
      kb_text_string: A [kb_num, max_kb_text_len] string tensor.
      kb_text_length: A [kb_num] int tensor.
      kb_mask: A [sent_num, kb_num] boolean tensor.
    """
    options = self._options

    # De-duplicate the kb ids within the example.
    #   kb_num shape = []; kb_ids shape = [kb_num]
    #   dup_kb_ids_flattened shape = [sent_num * dup_kb_num]

    with tf.name_scope('dedup_kb_id'):

      dup_kb_ids = self._query_to_id_table.lookup(query)
      dup_kb_ids_flattened = tf.sparse_tensor_to_dense(
          tf.strings.split(tf.reshape(dup_kb_ids, [-1]), sep=' '),
          default_value='')

      sent_num = tf.shape(query)[0]
      dup_kb_num = tf.shape(tf.reshape(dup_kb_ids_flattened, [sent_num, -1]))[1]

      non_empty_dup_kb_ids_flattened = tf.boolean_mask(
          dup_kb_ids_flattened, tf.not_equal(dup_kb_ids_flattened, ''))

      kb_ids, _ = tf.unique(non_empty_dup_kb_ids_flattened)
      kb_ids = kb_ids[:options.max_comments_per_image]
      kb_num = tf.shape(kb_ids)[0]

    # Retrieval contents.
    #   kb_text shape = [kb_num], full sentence.
    #   kb_text_string shape = [kb_num, max_kb_len], tokenized.
    #   kb_text_length shape = [kb_num].

    with tf.name_scope('retrieval'):

      kb_text = self._id_to_comment_table.lookup(kb_ids)
      kb_text_string = tf.sparse_tensor_to_dense(
          tf.strings.split(kb_text, sep=' '), default_value='')
      kb_text_string = kb_text_string[:, :options.max_tokens_to_keep]
      kb_text_length = tf.reduce_sum(
          tf.cast(tf.not_equal(kb_text_string, ''), dtype=tf.int32), axis=-1)

    # KB-query assignment.
    #   kb_mask shape = [sent_num, kb_num].

    with tf.name_scope('kb_assignment'):

      # kb_mask_flattened shape = [sent_num * dup_kb_num, kb_num]

      kb_mask_flattened = tf.equal(
          tf.expand_dims(dup_kb_ids_flattened, -1), tf.expand_dims(kb_ids, 0))

      kb_mask = tf.reshape(kb_mask_flattened, [sent_num, dup_kb_num, kb_num])
      kb_mask = tf.to_float(tf.reduce_any(kb_mask, axis=1))

    # Mask out query words in the retrieved contents.
    #   query: A [sent_num, query_num] string tensor.

    if options.remove_query:
      with tf.name_scope('mask_query_words'):
        query_flattented = tf.boolean_mask(query, tf.not_equal(query, ''))
        uniq_query, _ = tf.unique(query_flattented)

        equal_mask = tf.equal(
            tf.expand_dims(kb_text_string, axis=-1),
            tf.expand_dims(tf.expand_dims(uniq_query, 0), 0))
        not_equal_mask = tf.logical_not(tf.reduce_any(equal_mask, axis=-1))

        kb_text_string = tf.where(
            not_equal_mask, kb_text_string,
            tf.fill(dims=tf.shape(kb_text_string), value='[query]'))

    return (kb_num, kb_ids, kb_text_string, kb_text_length, kb_mask)


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.AdsReader):
    raise ValueError('options has to be an instance of Reader.')

  def _parse_fn(example):
    """Parses tf.Example proto.

    Args:
      example: a tf.Example proto.

    Returns:
      feature_dict: a dict mapping from names to tensors.
    """
    feature_dict = {}
    parsed = tf.parse_single_example(
        example, {
            TFExampleDataFields.image_id: tf.FixedLenFeature([], tf.int64),
            TFExampleDataFields.proposal_feature: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.proposal_text: tf.VarLenFeature(tf.string),
            TFExampleDataFields.proposal_ymin: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.proposal_xmin: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.proposal_ymax: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.proposal_xmax: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.slogan_text: tf.VarLenFeature(tf.string),
            TFExampleDataFields.slogan_ymin: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.slogan_xmin: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.slogan_ymax: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.slogan_xmax: tf.VarLenFeature(tf.float32),
            TFExampleDataFields.groundtruth_text: tf.VarLenFeature(tf.string),
            TFExampleDataFields.question_text: tf.VarLenFeature(tf.string),
        })

    # Image Id.

    feature_dict[InputDataFields.image_id] = (
        parsed[TFExampleDataFields.image_id])

    # Image proposals.

    proposal_feature = tf.reshape(
        tf.sparse_tensor_to_dense(parsed[TFExampleDataFields.proposal_feature]),
        [-1, options.feature_dimensions])
    feature_dict[InputDataFields.proposal_feature] = proposal_feature

    feature_dict[InputDataFields.proposal_box] = (
        tf.contrib.slim.tfexample_decoder.BoundingBox(
            prefix=TFExampleDataFields.proposal_box).tensors_to_item(parsed))

    (feature_dict[InputDataFields.proposal_num],
     feature_dict[InputDataFields.proposal_text_string],
     feature_dict[InputDataFields.proposal_text_length]
    ) = _decode_text_string_and_length(
        tf.sparse_tensor_to_dense(
            parsed[TFExampleDataFields.proposal_text], default_value=''))

    # Proposal labels.

    proposal_text_string = feature_dict[InputDataFields.proposal_text_string]
    uniq_label = tf.boolean_mask(
        proposal_text_string,
        tf.logical_and(
            tf.not_equal(proposal_text_string, 'null'),
            tf.not_equal(proposal_text_string, '')))
    uniq_label, _ = tf.unique(uniq_label)
    label_proposal_mask = tf.equal(
        tf.expand_dims(proposal_text_string, -1),
        tf.expand_dims(tf.expand_dims(uniq_label, 0), 0))
    label_proposal_mask = tf.to_float(
        tf.reduce_any(label_proposal_mask, axis=1))
    feature_dict[InputDataFields.proposal_label_num] = tf.shape(uniq_label)[0]
    feature_dict[InputDataFields.proposal_label_text] = uniq_label
    feature_dict[InputDataFields.proposal_label_mask] = label_proposal_mask

    # Text slogans.

    slogan_box = tf.contrib.slim.tfexample_decoder.BoundingBox(
        prefix=TFExampleDataFields.slogan_box).tensors_to_item(parsed)
    feature_dict[InputDataFields.slogan_box] = (
        slogan_box[:options.max_slogan_num, :])

    (feature_dict[InputDataFields.slogan_num],
     feature_dict[InputDataFields.slogan_text_string],
     feature_dict[InputDataFields.slogan_text_length]
    ) = _decode_text_string_and_length(
        tf.sparse_tensor_to_dense(
            parsed[TFExampleDataFields.slogan_text], default_value=''),
        max_string_num=options.max_slogan_num,
        max_string_len=options.max_slogan_len)

    # Annotations.

    (feature_dict[InputDataFields.groundtruth_num],
     feature_dict[InputDataFields.groundtruth_text_string],
     feature_dict[InputDataFields.groundtruth_text_length]
    ) = _decode_text_string_and_length(
        tf.sparse_tensor_to_dense(
            parsed[TFExampleDataFields.groundtruth_text], default_value=''),
        max_string_num=options.max_statement_num,
        max_string_len=options.max_statement_len)

    (feature_dict[InputDataFields.question_num],
     feature_dict[InputDataFields.question_text_string],
     feature_dict[InputDataFields.question_text_length]
    ) = _decode_text_string_and_length(
        tf.sparse_tensor_to_dense(
            parsed[TFExampleDataFields.question_text], default_value=''),
        max_string_num=options.max_statement_num,
        max_string_len=options.max_statement_len)

    # Query knowlege base to expand slogans.

    knowledge_base = KnowledgeBase(options.knowledge_base_config)

    (feature_dict[InputDataFields.slogan_kb_num],
     feature_dict[InputDataFields.slogan_kb_ids],
     feature_dict[InputDataFields.slogan_kb_text_string],
     feature_dict[InputDataFields.slogan_kb_text_length],
     feature_dict[InputDataFields.slogan_kb_mask]
    ) = knowledge_base.process_batch_query(
        feature_dict[InputDataFields.slogan_text_string])

    return feature_dict

  def _input_fn():
    """Returns a python dictionary.

    Returns:
      a dataset that can be fed to estimator.
    """
    input_pattern = [elem for elem in options.input_pattern]
    files = tf.data.Dataset.list_files(
        input_pattern, shuffle=options.is_training)
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=options.interleave_cycle_length)
    dataset = dataset.map(
        map_func=_parse_fn, num_parallel_calls=options.map_num_parallel_calls)
    if options.is_training:
      if options.cache == 'MEMORY':
        dataset = dataset.cache()
      elif options.cache:
        dataset = dataset.cache(filename=options.cache)
    if options.is_training:
      dataset = dataset.repeat().shuffle(options.shuffle_buffer_size)

    padded_shapes = {
        InputDataFields.image_id: [],
        InputDataFields.proposal_num: [],
        InputDataFields.proposal_feature: [None, options.feature_dimensions],
        InputDataFields.proposal_box: [None, 4],
        InputDataFields.proposal_text_string: [None, None],
        InputDataFields.proposal_text_length: [None],
        InputDataFields.slogan_num: [],
        InputDataFields.slogan_box: [None, 4],
        InputDataFields.slogan_text_string: [None, None],
        InputDataFields.slogan_text_length: [None],
        InputDataFields.groundtruth_num: [],
        InputDataFields.groundtruth_text_string: [None, None],
        InputDataFields.groundtruth_text_length: [None],
        InputDataFields.question_num: [],
        InputDataFields.question_text_string: [None, None],
        InputDataFields.question_text_length: [None],
        InputDataFields.proposal_label_num: [],
        InputDataFields.proposal_label_text: [None],
        InputDataFields.proposal_label_mask: [None, None],
        InputDataFields.slogan_kb_num: [],
        InputDataFields.slogan_kb_ids: [None],
        InputDataFields.slogan_kb_text_string: [None, None],
        InputDataFields.slogan_kb_text_length: [None],
        InputDataFields.slogan_kb_mask: [None, None],
    }

    dataset = dataset.padded_batch(
        options.batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
