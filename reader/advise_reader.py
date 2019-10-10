from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from core import utils
from core.standard_fields import InputDataFields
from core.standard_fields import TFExampleDataFields
from protos import reader_pb2

_OP_PARSE_SINGLE_EXAMPLE = 'reader/op_parse_single_example'
_OP_DECODE_IMAGE = 'reader/op_decode_image'
_OP_DECODE_ROI = 'reader/op_decode_proposal'
_OP_DECODE_SLOGAN = 'reader/op_decode_slogan'
_OP_DECODE_GROUNDTRUTH = 'reader/op_decode_groundtruth'
_OP_DECODE_QUESTION = 'reader/op_decode_question'
_OP_DECODE_BOX = 'reader/op_decode_box'


class KnowledgeBaseDict(object):

  def __init__(self,
               query_to_id_file,
               id_to_comment_file,
               remove_query=False,
               max_comments_per_image=10,
               max_tokens_to_keep=20):

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

    self._query_to_id_table = create_table_helper(query_to_id_file)
    self._id_to_comment_table = create_table_helper(id_to_comment_file)
    self._remove_query = remove_query
    self._max_comments_per_image = max_comments_per_image
    self._max_tokens_to_keep = max_tokens_to_keep

  def process_batch_query(self, query):
    """Queries knowledge base to get comments.
    Args:
      query: A [batch, max_sent_num, max_sent_len] string tensor.

    Returns:
      uniq_kb_ids: A [batch, max_uniq_kb_ids_size] string tensor, each element
        denotes an entry in the knowledge base subset.
      uniq_kb_content_strings: A [batch, max_uniq_kb_ids_size, 
        max_kb_content_len] string tensor, each element denotes an retrieved
        content in the knowledge base subset.
      uniq_kb_content_lengths: A [batch, max_uniq_kb_ids_size] int tensor.
      kb_mask: A [batch, max_sent_num, max_uniq_kb_ids_size] boolean tensor.
    """
    # Get the knowledge record associated with each statement.
    #   kb_ids shape = [batch, sent_num, kb_id_num]

    with tf.name_scope('associate_kb_to_slogan'):

      # Query to get kb_ids.
      #   kb_ids shape = [batch, sent_num, query_num]

      kb_ids = self._query_to_id_table.lookup(query)

      # Split the kb_ids.
      #   kb_ids shape = [batch, sent_num, kb_id_num]

      old_shape = tf.shape(kb_ids)[:2]  # [batch, sent_num]
      kb_ids_flattened = tf.strings.split(tf.reshape(kb_ids, [-1]), sep=' ')
      kb_ids_flattened = tf.sparse_tensor_to_dense(
          kb_ids_flattened, default_value='')

      kb_ids = tf.reshape(kb_ids_flattened, tf.concat([old_shape, [-1]], 0))

    # Create a knowledge base subset for each image.
    #   uniq_kb_ids shape = [batch, max_uniq_kb_ids_size]
    #   uniq_kb_ids_num shape = [batch]

    with tf.name_scope('create_uniq_kb_ids'):

      # Create a kb subset per image, require the batch to be static.

      batch, sent_num, kb_id_num = utils.get_tensor_shape(kb_ids)

      uniq_kb_ids, uniq_kb_ids_size = [], []

      for kb_ids_per_image in tf.unstack(kb_ids, axis=0):

        # kb_ids_per_image shape = [sent_num, kb_id_num]

        kb_ids_per_image_flattened = tf.reshape(kb_ids_per_image, [-1])
        kb_ids_per_image_flattened = tf.boolean_mask(
            kb_ids_per_image_flattened,
            tf.not_equal(kb_ids_per_image_flattened, ''),
            name='select_non_empty_kb_ids')

        uniq_kb_ids_per_image, _ = tf.unique(kb_ids_per_image_flattened)
        uniq_kb_ids_per_image = uniq_kb_ids_per_image[:self.
                                                      _max_comments_per_image]

        uniq_kb_ids.append(uniq_kb_ids_per_image)
        uniq_kb_ids_size.append(tf.shape(uniq_kb_ids_per_image)[0])

      # Pad the uniq_kb_ids to the same size.
      #   uniq_kb_ids shape = [batch, max_uniq_kb_ids_size]

      max_uniq_kb_ids_size = tf.reduce_max(tf.stack(uniq_kb_ids_size, 0))

      for i in range(len(uniq_kb_ids)):
        uniq_kb_ids[i] = tf.concat([
            uniq_kb_ids[i],
            tf.fill(
                tf.expand_dims(max_uniq_kb_ids_size - uniq_kb_ids_size[i], 0),
                value='')
        ], 0)
      uniq_kb_ids = tf.stack(uniq_kb_ids, axis=0)
      uniq_kb_ids_num = tf.reduce_sum(
          tf.cast(tf.not_equal(uniq_kb_ids, ''), tf.int32), axis=-1)

    # Retrieve contents using the uniq_kb_ids.
    #   uniq_kb_contents shape = [batch, max_uniq_kb_ids_size, max_kb_content_len].

    with tf.name_scope('retrieve_and_tokenize_knowledge'):

      # Retrieve knowledge.
      #   uniq_kb_contents shape = [batch, max_uniq_kb_ids_size].

      uniq_kb_contents = self._id_to_comment_table.lookup(uniq_kb_ids)

      old_shape = tf.shape(uniq_kb_contents)  # [batch, max_uniq_kb_ids_siz]

      uniq_kb_contents_flattened = tf.strings.split(
          tf.reshape(uniq_kb_contents, [-1]), sep=' ')
      uniq_kb_contents_flattened = tf.sparse_tensor_to_dense(
          uniq_kb_contents_flattened, default_value='')

      uniq_kb_contents = tf.reshape(uniq_kb_contents_flattened,
                                    tf.concat([old_shape, [-1]], 0))

      # Trim long descriptions.

      uniq_kb_content_strings = uniq_kb_contents[:, :, :self.
                                                 _max_tokens_to_keep]
      uniq_kb_content_lengths = tf.reduce_sum(
          tf.cast(tf.not_equal(uniq_kb_content_strings, ''), dtype=tf.int32),
          axis=-1)

    # KB record assignment.
    #   uniq_kb_ids shape = [batch, max_uniq_kb_ids_size]
    #   kb_ids shape = [batch, sent_num, kb_id_num]
    #   kb_mask shape = [batch, sent_num, max_uniq_kb_ids_size]

    with tf.name_scope('assign_kb_record'):
      # kb_mask_flattened shape = [batch, sent_num * kb_id_num, max_uniq_kb_ids_size]

      kb_mask_flattened = tf.logical_and(
          tf.equal(
              tf.reshape(kb_ids, [batch, -1, 1]), tf.expand_dims(
                  uniq_kb_ids, 1)),
          tf.expand_dims(tf.not_equal(uniq_kb_ids, ''), 1))

      # kb_mask shape = [batch, sent_num, kb_id_num, max_uniq_kb_ids_size]

      kb_mask = tf.reshape(kb_mask_flattened,
                           [batch, sent_num, kb_id_num, max_uniq_kb_ids_size])

      kb_mask = tf.reduce_any(kb_mask, axis=2)

    # Mask out query words in the retrieved contents.

    if self._remove_query:
      with tf.name_scope('mask_query_words'):
        masked_uniq_kb_content_strings = []
        for query_per_image, uniq_kb_content_strings_per_image in zip(
            tf.unstack(query, axis=0), tf.unstack(uniq_kb_content_strings, axis=0)):
          query_per_image_flattented = tf.boolean_mask(
              query_per_image,
              tf.not_equal(query_per_image, ''),
              name='select_non_empty_query_words')
          uniq_query_words_per_image, _ = tf.unique(query_per_image_flattented)

          equal_mask = tf.equal(
              tf.expand_dims(uniq_kb_content_strings_per_image, axis=-1),
              tf.reshape(uniq_query_words_per_image, [1, 1, -1]))
          not_equal_mask = tf.logical_not(tf.reduce_any(equal_mask, axis=-1))
          masked_uniq_kb_content_strings.append(
              tf.where(
                  not_equal_mask, uniq_kb_content_strings_per_image,
                  tf.fill(
                      dims=tf.shape(uniq_kb_content_strings_per_image),
                      value='[query]')))
        uniq_kb_content_strings = tf.stack(masked_uniq_kb_content_strings, axis=0)

    return (uniq_kb_ids, uniq_kb_ids_num, uniq_kb_content_strings,
            uniq_kb_content_lengths, kb_mask)


class TFExampleDataFields(object):
  """Names of the fields of the tf.train.Example."""
  image_id = "image_id"
  image_feature = 'feature/img/value'
  proposal_num = 'feature/roi/length'
  proposal_feature = 'feature/roi/value'

  slogan_box = 'ocr/bbox'
  slogan_box_ymin = 'ocr/bbox/ymin'
  slogan_box_xmin = 'ocr/bbox/xmin'
  slogan_box_ymax = 'ocr/bbox/ymax'
  slogan_box_xmax = 'ocr/bbox/xmax'
  slogan_text_string = 'ocr/text/string'
  slogan_text_offset = 'ocr/text/offset'
  slogan_text_length = 'ocr/text/length'

  groundtruth_text_string = 'label/text/string'
  groundtruth_text_offset = 'label/text/offset'
  groundtruth_text_length = 'label/text/length'
  question_text_string = 'question/text/string'
  question_text_offset = 'question/text/offset'
  question_text_length = 'question/text/length'
  proposal_box = 'roi/bbox'
  proposal_box_ymin = 'roi/bbox/ymin'
  proposal_box_xmin = 'roi/bbox/xmin'
  proposal_box_ymax = 'roi/bbox/ymax'
  proposal_box_xmax = 'roi/bbox/xmax'
  proposal_text_string = 'roi/text/string'
  proposal_text_offset = 'roi/text/offset'
  proposal_text_length = 'roi/text/length'


class InputDataFields(object):
  """Names of the input tensors."""
  image_id = 'image_id'
  image_feature = 'image_feature'
  proposal_num = 'proposal_num'
  proposal_box = 'proposal_box'
  proposal_feature = 'proposal_feature'
  proposal_text_string = 'proposal_text_string'
  proposal_text_length = 'proposal_text_length'
  slogan_num = 'slogan_num'
  slogan_box = 'slogan_box'
  slogan_text_string = 'slogan_text_string'
  slogan_text_length = 'slogan_text_length'
  groundtruth_num = 'groundtruth_num'
  groundtruth_text_string = 'groundtruth_text_string'
  groundtruth_text_length = 'groundtruth_text_length'
  question_num = 'question_num'
  question_text_string = 'question_text_string'
  question_text_length = 'question_text_length'

  slogan_kb_ids = 'sparql_kb_ids'
  slogan_kb_num = 'sparql_kb_num'
  slogan_kb_text_string = 'sparql_kb_text_string'
  slogan_kb_text_length = 'sparql_kb_text_length'
  slogan_kb_mask = 'sparql_slogan_kb_mask'


def _parse_texts(tokens,
                 offsets,
                 lengths,
                 max_num_texts=30,
                 max_text_length=100):
  """Parses and pads texts.

  Args:
    tokens: a [num_tokens] tf.string tensor denoting token buffer.
    offsets: a [num_texts] tf.int64 tensor, denoting the offset of each
      text in the token buffer.
    lengths: a [num_texts] tf.int64 tensor, denoting the length of each
      text.

  Returns:
    num_texts: number of texts after padding.
    text_strings: [num_texts, max_text_length] tf.string tensor.
    text_lengths: [num_texts] tf.int64 tensor.
  """
  max_text_length0 = tf.maximum(tf.reduce_max(lengths), 0)
  max_text_length = tf.minimum(
      tf.cast(max_text_length, tf.int64), max_text_length0)

  num_offsets = tf.shape(offsets)[0]
  num_lengths = tf.shape(lengths)[0]

  assert_op = tf.Assert(
      tf.equal(num_offsets, num_lengths),
      ["Not equal: num_offsets and num_lengths", num_offsets, num_lengths])

  with tf.control_dependencies([assert_op]):
    num_texts = num_offsets

    i = tf.constant(0)
    text_strings = tf.fill(tf.stack([0, max_text_length], axis=0), "")
    text_lengths = tf.constant(0, dtype=tf.int64, shape=[0])

    def _body(i, text_strings, text_lengths):
      """Executes the while loop body.

      Note, this function trims or pads texts to make them the same lengths.

      Args:
        i: index of both offsets/lengths tensors.
        text_strings: aggregated text strings tensor.
        text_lengths: aggregated text lengths tensor.
      """
      offset = offsets[i]
      length = tf.minimum(lengths[i], max_text_length)

      pad = tf.fill(tf.expand_dims(max_text_length - length, axis=0), "")
      text = tokens[offset:offset + length]
      text = tf.concat([text, pad], axis=0)
      text_strings = tf.concat([text_strings, tf.expand_dims(text, 0)], axis=0)
      text_lengths = tf.concat(
          [text_lengths, tf.expand_dims(length, 0)], axis=0)
      return i + 1, text_strings, text_lengths

    cond = lambda i, unused_strs, unused_lens: tf.less(i, num_texts)
    (_, text_strings, text_lengths) = tf.while_loop(
        cond,
        _body,
        loop_vars=[i, text_strings, text_lengths],
        shape_invariants=[
            i.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None])
        ])

  num_texts = tf.minimum(max_num_texts, num_texts)
  text_strings = text_strings[:num_texts, :]
  text_lengths = text_lengths[:num_texts]

  return num_texts, text_strings, text_lengths


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.AdViSEReader):
    raise ValueError('options has to be an instance of Reader.')

  def _parse_fn(example):
    """Parses tf::Example proto.

    Args:
      example: a tf::Example proto.

    Returns:
      feature_dict: a dict mapping from names to tensors.
    """
    example_fmt = {
        TFExampleDataFields.image_id:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.image_feature:
        tf.FixedLenFeature([options.feature_dimensions], tf.float32),
        TFExampleDataFields.proposal_num:
        tf.FixedLenFeature([], tf.int64),
        TFExampleDataFields.proposal_feature:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.slogan_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.slogan_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.proposal_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.proposal_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.proposal_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.groundtruth_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.groundtruth_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.groundtruth_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_string:
        tf.VarLenFeature(tf.string),
        TFExampleDataFields.question_text_offset:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.question_text_length:
        tf.VarLenFeature(tf.int64),
        TFExampleDataFields.proposal_box_ymin:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_xmin:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_ymax:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.proposal_box_xmax:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_box_ymin:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_box_xmin:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_box_ymax:
        tf.VarLenFeature(tf.float32),
        TFExampleDataFields.slogan_box_xmax:
        tf.VarLenFeature(tf.float32),
    }
    parsed = tf.parse_single_example(
        example, example_fmt, name=_OP_PARSE_SINGLE_EXAMPLE)

    feature_dict = {
        InputDataFields.image_id: parsed[TFExampleDataFields.image_id],
    }

    # Decode image feature.

    with tf.name_scope(_OP_DECODE_IMAGE):
      (feature_dict[InputDataFields.image_feature],
       feature_dict[InputDataFields.proposal_num]) = (
           parsed[TFExampleDataFields.image_feature],
           parsed[TFExampleDataFields.proposal_num])
      feature_dict[InputDataFields.proposal_feature] = tf.reshape(
          tf.sparse_tensor_to_dense(
              parsed[TFExampleDataFields.proposal_feature]),
          [-1, options.feature_dimensions])

    tuples = [
        (_OP_DECODE_ROI, TFExampleDataFields.proposal_text_string,
         TFExampleDataFields.proposal_text_offset,
         TFExampleDataFields.proposal_text_length, 'tmp',
         InputDataFields.proposal_text_string,
         InputDataFields.proposal_text_length),
        (_OP_DECODE_SLOGAN, TFExampleDataFields.slogan_text_string,
         TFExampleDataFields.slogan_text_offset,
         TFExampleDataFields.slogan_text_length, InputDataFields.slogan_num,
         InputDataFields.slogan_text_string,
         InputDataFields.slogan_text_length),
        (_OP_DECODE_GROUNDTRUTH, TFExampleDataFields.groundtruth_text_string,
         TFExampleDataFields.groundtruth_text_offset,
         TFExampleDataFields.groundtruth_text_length,
         InputDataFields.groundtruth_num,
         InputDataFields.groundtruth_text_string,
         InputDataFields.groundtruth_text_length),
        (_OP_DECODE_QUESTION, TFExampleDataFields.question_text_string,
         TFExampleDataFields.question_text_offset,
         TFExampleDataFields.question_text_length, InputDataFields.question_num,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
    ]

    for (name_scope, input_string_field, input_offset_field, input_length_field,
         output_size_field, output_string_field, output_length_field) in tuples:
      with tf.name_scope(name_scope):
        max_text_length = options.max_text_length
        max_num_texts = options.max_num_texts
        if name_scope == _OP_DECODE_QUESTION:
          max_num_texts = 10000
        if name_scope == _OP_DECODE_GROUNDTRUTH:
          max_text_length = options.max_stmt_length

        (feature_dict[output_size_field], feature_dict[output_string_field],
         feature_dict[output_length_field]) = _parse_texts(
             tokens=tf.sparse_tensor_to_dense(
                 parsed[input_string_field], default_value=""),
             offsets=tf.sparse_tensor_to_dense(
                 parsed[input_offset_field], default_value=0),
             lengths=tf.sparse_tensor_to_dense(
                 parsed[input_length_field], default_value=0),
             max_num_texts=max_num_texts,
             max_text_length=max_text_length)

    # Query knowledge dict to get the dbpedia contents.

    kb_dict = KnowledgeBaseDict(
        options.knowledge_query_to_id_file,
        options.knowledge_id_to_comment_file,
        remove_query=options.knowledge_remove_query,
        max_comments_per_image=options.knowledge_max_comments_per_image,
        max_tokens_to_keep=options.knowledge_max_tokens_to_keep)

    (kb_ids, kb_num, kb_content_string, kb_content_length,
     kb_slogan_mask) = kb_dict.process_batch_query(
         tf.expand_dims(
             feature_dict[InputDataFields.slogan_text_string], axis=0))

    feature_dict[InputDataFields.slogan_kb_ids] = tf.squeeze(kb_ids, 0)
    feature_dict[InputDataFields.slogan_kb_num] = tf.squeeze(kb_num, 0)
    feature_dict[InputDataFields.slogan_kb_text_string] = tf.squeeze(
        kb_content_string, 0)
    feature_dict[InputDataFields.slogan_kb_text_length] = tf.squeeze(
        kb_content_length, 0)
    feature_dict[InputDataFields.slogan_kb_mask] = tf.to_float(
        tf.squeeze(kb_slogan_mask, 0))

    with tf.name_scope(_OP_DECODE_BOX):
      bbox_decoder = tf.contrib.slim.tfexample_decoder.BoundingBox(
          prefix=TFExampleDataFields.proposal_box + '/')
      feature_dict[InputDataFields.proposal_box] = bbox_decoder.tensors_to_item(
          parsed)

      bbox_decoder = tf.contrib.slim.tfexample_decoder.BoundingBox(
          prefix=TFExampleDataFields.slogan_box + '/')
      feature_dict[InputDataFields.slogan_box] = bbox_decoder.tensors_to_item(
          parsed)[:options.max_num_texts, :]

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
        InputDataFields.image_feature: [options.feature_dimensions],
        InputDataFields.proposal_num: [],
        InputDataFields.proposal_feature: [None, options.feature_dimensions],
        InputDataFields.proposal_box: [None, 4],
        InputDataFields.slogan_box: [None, 4],
        InputDataFields.slogan_kb_ids: [None],
        InputDataFields.slogan_kb_num: [],
        InputDataFields.slogan_kb_text_string: [None, None],
        InputDataFields.slogan_kb_text_length: [None],
        InputDataFields.slogan_kb_mask: [None, None],
    }

    tuples = [
        (_OP_DECODE_ROI, 'tmp', InputDataFields.proposal_text_string,
         InputDataFields.proposal_text_length),
        (_OP_DECODE_SLOGAN, InputDataFields.slogan_num,
         InputDataFields.slogan_text_string,
         InputDataFields.slogan_text_length),
        (_OP_DECODE_GROUNDTRUTH, InputDataFields.groundtruth_num,
         InputDataFields.groundtruth_text_string,
         InputDataFields.groundtruth_text_length),
        (_OP_DECODE_QUESTION, InputDataFields.question_num,
         InputDataFields.question_text_string,
         InputDataFields.question_text_length),
    ]

    for name_scope, output_size_field, output_string_field, output_length_field in tuples:
      padded_shapes[output_size_field] = []
      padded_shapes[output_string_field] = [None, None]
      padded_shapes[output_length_field] = [None]

    dataset = dataset.padded_batch(
        options.batch_size, padded_shapes=padded_shapes, drop_remainder=True)
    dataset = dataset.prefetch(options.prefetch_buffer_size)
    return dataset

  return _input_fn
