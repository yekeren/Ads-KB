from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import cv2
import nltk
import collections

import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('image_data_path', '',
                    'Path to the directory saving images in jpeg format.')

flags.DEFINE_integer('number_of_processes', 100, 'Number of processes.')

flags.DEFINE_integer('process_id', 0, 'Id of the current process.')

flags.DEFINE_string('tfrecord_output_path', '',
                    'Path to the directory saving output .tfrecord files.')

flags.DEFINE_string(
    'proposal_json_path', '',
    'Path to the directory saving proposal annotations in json format.')

flags.DEFINE_string(
    'proposal_feature_npy_path', '',
    'Path to the directory saving proposal features in npy format.')

flags.DEFINE_string(
    'slogan_json_path', '',
    'Path to the directory saving slogan annotations in json format.')

flags.DEFINE_string(
    'statement_json_path', '',
    'Path to the directory saving statement annotations in json format.')

FLAGS = flags.FLAGS

_tokenize = lambda x: nltk.word_tokenize(x.lower())


def _load_image_path_list(image_data_path):
  """Loads image paths from the image_data_path.

  Args:
    image_data_path: path to the directory saving images.

  Returns:
    examples: a list of (image_id, filename) tuples.
  """
  examples = []
  for dirpath, dirnames, filenames in os.walk(image_data_path):
    for filename in filenames:
      image_id = int(filename.split('.')[0])
      filename = os.path.join(dirpath, filename)
      examples.append((image_id, filename))
  return examples


def _decode_statement(data):
  """Decodes groundtruth annotations and candidate questions.

  Args:
    data: a python dict containing keys of `groundtruth_list` and `question_list`.

  Returns:
    groundtruth: A list of strings representing the annotated ground-truth.
    question: A list of strings representing the questions.
  """
  groundtruth = [
      ' '.join(_tokenize(x)).encode('utf8') for x in data['groundtruth_list']
  ]
  question = [
      ' '.join(_tokenize(x)).encode('utf8') for x in data['question_list']
  ]

  return {
      'groundtruth': groundtruth,
      'question': question,
  }


def _decode_box_and_text(data):
  """Decodes bounding boxes and texts associated with them.

  Note: text_string[text_offset[i]: text_offset[i] + text_length[i]] denotes the text for the `i`th box.

  Args:
    data: a python dict containing keys of `text` and `paragraphs`.

  Returns:
    ymin: a list of float, each represents a ymin coordinate of a box.
    xmin: a list of float, each represents a xmin coordinate of a box.
    ymax: a list of float, each represents a ymax coordinate of a box.
    xmax: a list of float, each represents a xmax coordinate of a box.
    text: a list of string, each denotes a description of a box.
  """
  ymin = []
  xmin = []
  ymax = []
  xmax = []
  text = []
  for paragraph in data['paragraphs']:
    ymin.append(paragraph['bounding_box']['ymin'])
    xmin.append(paragraph['bounding_box']['xmin'])
    ymax.append(paragraph['bounding_box']['ymax'])
    xmax.append(paragraph['bounding_box']['xmax'])
    tokens = _tokenize(paragraph['text'])
    text.append(' '.join(tokens).encode('utf8'))

  return {
      'ymin': ymin,
      'xmin': xmin,
      'ymax': ymax,
      'xmax': xmax,
      'text': text,
  }


def _add_box_and_text(tf_example, scope, data):
  """Adds box-level caption annotations to the tf.train.Example proto.

  Args:
    tf_example: an instance of tf.train.Example.
    scope: name in the tf.train.Example.
    data: 
  """
  feature_map = tf_example.features.feature

  for name in ['ymin', 'xmin', 'ymax', 'xmax']:
    feature_map[scope + '/bbox/' + name].float_list.CopyFrom(
        tf.train.FloatList(value=data[name]))

  feature_map[scope + '/bbox/text'].bytes_list.CopyFrom(
      tf.train.BytesList(value=data['text']))
  return tf_example


def _dict_to_tf_example(data):
  """Converts python dict to tf example.

  Args:
    data: the python dict returned by `_load_annotation`.

  Returns:
    tf_example: the tf.train.Example proto.
  """
  print(len(data['statement']['groundtruth_list']))
  # Add the image_id and proposal feature vectors.

  tf_example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image_id':
              tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[data['image_id']])),
              'proposal/bbox/feature':
              tf.train.Feature(
                  float_list=tf.train.FloatList(
                      value=data['proposal_feature'].flatten().tolist())),
          }))
  feature_map = tf_example.features.feature

  # Encode the proposal and slogan annotations.

  tf_example = _add_box_and_text(
      tf_example, scope='proposal', data=_decode_box_and_text(data['proposal']))

  tf_example = _add_box_and_text(
      tf_example, scope='slogan', data=_decode_box_and_text(data['slogan']))

  # Encode the statement annotations.

  statement_data = _decode_statement(data['statement'])

  feature_map['groundtruth/text'].bytes_list.CopyFrom(
      tf.train.BytesList(value=statement_data['groundtruth']))
  feature_map['question/text'].bytes_list.CopyFrom(
      tf.train.BytesList(value=statement_data['question']))

  return tf_example


def _load_annotation(image_id, image_filename, proposal_json_path,
                     proposal_feature_npy_path, slogan_json_path,
                     statement_json_path):
  """Loads the annotation for the image.

  Args:
    image_id: Numeric id of the image.
    image_filename: Path to the image file.
    proposal_json_path: Path to the proposal annotation.
    proposal_feature_npy_path: Path to the proposal features.
    slogan_json_path: Path to the slogan annotation.
    statement_json_path: Path to the statement annotation.

  Returns:
    a python dict containing the annotations.
  """

  def _load_npy(filename):
    """Loads data in npy format."""
    with open(filename, 'rb') as fid:
      return np.load(fid)

  def _load_json(filename):
    """Loads data in json format."""
    with open(filename, 'r') as fid:
      return json.load(fid)

  json_filename = '{}.json'.format(image_id)
  npy_filename = '{}.npy'.format(image_id)

  data = {}
  data['image_id'] = image_id
  data['proposal'] = _load_json(os.path.join(proposal_json_path, json_filename))
  data['proposal_feature'] = _load_npy(
      os.path.join(proposal_feature_npy_path, npy_filename))
  data['slogan'] = _load_json(os.path.join(slogan_json_path, json_filename))
  data['statement'] = _load_json(
      os.path.join(statement_json_path, json_filename))
  return data


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  examples = _load_image_path_list(FLAGS.image_data_path)
  tf.logging.info('Load %s examples.', len(examples))

  filename = FLAGS.tfrecord_output_path + '-%05d-of-%05d' % (
      FLAGS.process_id, FLAGS.number_of_processes)
  writer = tf.python_io.TFRecordWriter(filename)

  for index, (image_id, filename) in enumerate(examples):
    if index % FLAGS.number_of_processes == FLAGS.process_id:
      data = _load_annotation(image_id, filename, FLAGS.proposal_json_path,
                              FLAGS.proposal_feature_npy_path,
                              FLAGS.slogan_json_path, FLAGS.statement_json_path)

      tf_example = _dict_to_tf_example(data)
      writer.write(tf_example.SerializeToString())

    if index % 50 == 0:
      tf.logging.info('On image %i/%i', index, len(examples))

  writer.close()


if __name__ == '__main__':
  tf.app.run()
