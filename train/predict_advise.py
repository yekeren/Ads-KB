from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import numpy as np
import collections
import tensorflow as tf
from google.protobuf import text_format
from collections import Counter
from collections import defaultdict

from protos import pipeline_pb2
from train import trainer
from core.training_utils import save_model_if_it_is_better

from readers.ads_reader import InputDataFields

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('saved_ckpts_dir', '',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('qa_json_path', 'raw_data/qa.json/',
                    'Path to the qa annotation.')

flags.DEFINE_string('topic_json_input_path',
                    'raw_data/ads_test_annotations/Topics_test.json',
                    'Path to the qa annotation.')

flags.DEFINE_string('topic_namelist_txt_input_path', 'raw_data/topics_list.txt',
                    'Path to the qa annotation.')

#flags.DEFINE_string(
#    'proposal_json_path', 'raw_data/ads_wsod.json/nmsed',
#    'Path to the directory saving proposal annotations in json format.')
#
#flags.DEFINE_string(
#    'slogan_json_path', 'raw_data/ocr.json/',
#    'Path to the directory saving ocr annotations in json format.')

flags.DEFINE_string('json_output_path', 'raw_data/visl.json',
                    'Path to the output json path.')

flags.DEFINE_string('prediction_output_path', 'results.json',
                    'Path to the prediction results.')

flags.DEFINE_string('input_pattern', '', 'Path to the prediction results.')

flags.DEFINE_string('metrics_output_format', 'csv',
                    'Format to output the metrics.')

flags.DEFINE_bool('run_once', True, 'If true, run once.')

flags.DEFINE_string('eval_log_dir', '', 'Path to the eval log dir.')

flags.DEFINE_integer('max_steps', 0, 'Maximum training steps.')
flags.DEFINE_integer('eval_min_steps', 3000, 'Maximum training steps.')
flags.DEFINE_integer('number_of_eval_examples', 999999999999, 'Maximum training steps.')

FLAGS = flags.FLAGS

_UNCLEAR = 'unclear'

_FIELD_IMAGE_ID = 'image_id'
_FIELD_IMAGE_IDS_GATHERED = 'image_ids_gathered'
_FIELD_SIMILARITY = 'similarity'
_FIELD_ADJACENCY = 'adjacency'
_FIELD_ADJACENCY_LOGITS = 'adjacency_logits'

_PSAs = set([
    'environment', 'animal_right', 'human_right', 'safety',
    'smoking_alcohol_abuse', 'domestic_violence', 'self_esteem', 'political',
    'charities'
])


class Metrics(object):

  def __init__(self):
    self._metrics = defaultdict(list)

  def update(self, tag, values):
    self._metrics[tag].append(values)

  def report(self, tag, column):
    if not tag in self._metrics:
      return 0.0
    data = np.array(self._metrics[tag]).mean(axis=0)
    return data[column]


def _load_pipeline_proto(filename):
  """Loads pipeline proto from file.

  Args:
    filename: path to the pipeline config file.

  Returns:
    an instance of pipeline_pb2.Pipeline.
  """
  pipeline_proto = pipeline_pb2.Pipeline()
  with tf.gfile.GFile(filename, 'r') as fp:
    text_format.Merge(fp.read(), pipeline_proto)
  return pipeline_proto


def _load_topics(topic_json_input_path, topic_namelist_txt_input_path):
  """Loads topic annotations.

  Args:
    topic_json_input_path: Path to the directory saving topic annotations.
    topic_namelist_txt_input_path: Path to the file storing topic list.

  Returns:
    A python dict mapping from image ID to the topic annotations.
  """
  with open(topic_namelist_txt_input_path, 'r') as fid:
    topic_list = [line.strip('\n') for line in fid.readlines()]

  topics = dict(
      [(str(index + 1), topic) for index, topic in enumerate(topic_list)])

  topic_annotations = {}
  with open(topic_json_input_path, 'r') as fid:
    data = json.load(fid)
    for image_id, annotations in data.items():
      image_id = int(image_id.split('/')[1].split('.')[0])
      counter = Counter([
          topics.get(annotation, _UNCLEAR)
          for annotation in annotations
          if annotation in topics
      ])
      if len(counter):
        topic_annotations[image_id] = counter.most_common()[0][0]
      else:
        topic_annotations[image_id] = _UNCLEAR
  return topic_list, topic_annotations


def _revise_image_id(image_id):
  """Revises image id.

  Args:
    image_id: Image ID in numeric number format.

  Returns:
    Image ID in `subdir/filename` format.
  """
  if image_id >= 170000:
    image_id = '10/{}.png'.format(image_id)
  else:
    image_id = '{}/{}.jpg'.format(image_id % 10, image_id)
  return image_id


def _load_json(filename):
  """Loads data in json format.

  Args:
    filename: Path to the json file.

  Returns:
    a python dict representing the json object.
  """
  with open(filename, 'r') as fid:
    return json.load(fid)


def _boxes_to_json_array(boxes):
  array = []
  for box in boxes:
    ymin, xmin, ymax, xmax = [round(float(x), 3) for x in box]
    array.append({'ymin': ymin, 'xmin': xmin, 'ymax': ymax, 'xmax': xmax})
  return array


def _varlen_strings_to_json_array(strings, lengths):
  array = []
  for string, length in zip(strings, lengths):
    tokens = [x.decode('utf8') for x in string[:length]]
    array.append(' '.join(tokens))
  return array


def _update_metrics(groundtruth_list, prediction_list, topic, metrics):
  """Updtes the metrics.

  Args:
    metrics: An instance of collections.defaultdict.
    groundtruth_list: A list of strings, which are the groundtruth annotations.
    prediction_list: A list of predictions.

  Returns:
    updated metrics.
  """
  if not groundtruth_list:
    return metrics

  bingo = 1.0 if prediction_list[0] in groundtruth_list else 0.0
  ranks = [1.0 + prediction_list.index(x) for x in groundtruth_list]

  min_rank = np.min(ranks)
  avg_rank = np.mean(ranks)
  med_rank = np.median(ranks)

  metrics.update(
      tag='general_micro', values=[bingo, min_rank, avg_rank, med_rank])
  metrics.update(tag=topic, values=[bingo, min_rank, avg_rank, med_rank])
  if topic in _PSAs:
    metrics.update(
        tag='psa_micro', values=[bingo, min_rank, avg_rank, med_rank])
  elif topic != _UNCLEAR:
    metrics.update(
        tag='product_micro', values=[bingo, min_rank, avg_rank, med_rank])
  return metrics


def _run_prediction(pipeline_proto,
                    topic_list,
                    topic_data,
                    checkpoint_path=None):
  """Runs the prediction.

  Args:
    pipeline_proto: an instance of pipeline_pb2.Pipeline.
  """
  results = {}
  metrics = Metrics()

  for example_index, example in enumerate(
      trainer.predict(pipeline_proto, checkpoint_path=checkpoint_path)):

    # Compute the metrics.
    image_id = example['image_id'][0]

    annotation = _load_json(
        os.path.join(FLAGS.qa_json_path, '{}.json'.format(image_id)))
    (groundtruth_list, question_list) = (annotation['groundtruth_list'],
                                         annotation['question_list'])
    prediction_list = [
        question_list[i] for i in example['similarity'][0].argsort()[::-1]
    ]

    topic = topic_data[image_id]
    _update_metrics(groundtruth_list, prediction_list, topic, metrics)

    # Create the result entry to write into the .json file.

    results[_revise_image_id(image_id)] = [
        question_list[index]
        for index in np.argsort(example['similarity'][0])[::-1]
    ]

    if example_index % 100 == 0:
      tf.logging.info('On image %i', example_index)

    if example_index + 1 >= FLAGS.number_of_eval_examples:
      break

    # Create json visualization.

    (
        adjacency,
        adjacency_logits,
        similarity,
        proposal_num,
        proposal_box,
        proposal_strings,
        proposal_lengths,
        slogan_num,
        slogan_box,
        slogan_strings,
        slogan_lengths,
        slogan_kb_num,
        slogan_kb_strings,
        slogan_kb_lengths,
    ) = (example[_FIELD_ADJACENCY][0], example[_FIELD_ADJACENCY_LOGITS][0],
         example[_FIELD_SIMILARITY][0], example[InputDataFields.proposal_num][0],
         example[InputDataFields.proposal_box][0],
         example[InputDataFields.proposal_text_string][0],
         example[InputDataFields.proposal_text_length][0],
         example[InputDataFields.slogan_num][0],
         example[InputDataFields.slogan_box][0],
         example[InputDataFields.slogan_text_string][0],
         example[InputDataFields.slogan_text_length][0],
         example[InputDataFields.slogan_kb_num][0],
         example[InputDataFields.slogan_kb_text_string][0],
         example[InputDataFields.slogan_kb_text_length][0])

    # Results for visualization.

    with open(
        os.path.join(FLAGS.json_output_path, '{}.json'.format(image_id)),
        'w') as fid:
      json_data = {
          'image_id':
          int(image_id),
          'proposal_num':
          int(proposal_num),
          'proposal_boxes':
          _boxes_to_json_array(proposal_box),
          'proposal_labels':
          _varlen_strings_to_json_array(proposal_strings, proposal_lengths),
          'slogan_num':
          int(slogan_num),
          'slogan_boxes':
          _boxes_to_json_array(slogan_box),
          'slogan_labels':
          _varlen_strings_to_json_array(slogan_strings, slogan_lengths),
          'slogan_kb_num':
          int(slogan_kb_num),
          'slogan_kb_labels':
          _varlen_strings_to_json_array(slogan_kb_strings, slogan_kb_lengths),
          'adjacency': [[round(float(x), 2) for x in row] for row in adjacency],
          'adjacency_logits':
          [[round(float(x), 2) for x in row] for row in adjacency_logits],
          'predictions':
          results[_revise_image_id(image_id)],
          'annotations':
          groundtruth_list
      }

      fid.write(json.dumps(json_data, indent=2))

  # Metrics.

  accuracy_list, minrank_list = [], []
  for topic in topic_list:
    accuracy_list.append(metrics.report(tag=topic, column=0))
    minrank_list.append(metrics.report(tag=topic, column=1))
  accuracy_list = [round(x, 3) for x in accuracy_list]
  minrank_list = [round(x, 3) for x in minrank_list]

  accuracy_product = np.mean(accuracy_list[:-9])
  accuracy_psa = np.mean(accuracy_list[-9:])
  minrank_product = np.mean(minrank_list[:-9])
  minrank_psa = np.mean(minrank_list[-9:])

  tf.logging.info('-' * 128)
  tf.logging.info('accuracy: product=%.3lf, psa=%.3lf', accuracy_product,
                  accuracy_psa)
  tf.logging.info('minrank: product=%.3lf, psa=%.3lf', minrank_product,
                  minrank_psa)

  # Results to be submitted.

  with open(FLAGS.prediction_output_path, 'w') as fid:
    fid.write(json.dumps(results, indent=2))

  # Test ids to be exported.

  image_ids = [int(x.split('/')[1].split('.')[0]) for x in results.keys()]
  with open('image_id.txt', 'w') as fid:
    fid.write(json.dumps(image_ids, indent=2))

  results = {
      'accuracy/macro': np.mean(accuracy_list),
      'accuracy/micro': metrics.report(tag='general_micro', column=0),
      'accuracy/product_macro': accuracy_product,
      'accuracy/product_micro': metrics.report(tag='product_micro', column=0),
      'accuracy/psa_macro': accuracy_psa,
      'accuracy/psa_micro': metrics.report(tag='psa_micro', column=0),
      'minrank/macro': np.mean(minrank_list),
      'minrank/micro': metrics.report(tag='general_micro', column=1),
      'minrank/product_macro': minrank_product,
      'minrank/product_micro': metrics.report(tag='product_micro', column=1),
      'minrank/psa_macro': minrank_psa,
      'minrank/psa_micro': metrics.report(tag='psa_micro', column=1),
  }
  for topic, accuracy in zip(topic_list, accuracy_list):
    results['accuracy_per_topic/{}'.format(topic)] = accuracy
  return results


def main(_):
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  if FLAGS.model_dir:
    pipeline_proto.model_dir = FLAGS.model_dir
    tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

  if FLAGS.input_pattern:
    while len(pipeline_proto.eval_reader.ads_reader.input_pattern) > 0:
      pipeline_proto.eval_reader.ads_reader.input_pattern.pop()
    pipeline_proto.eval_reader.ads_reader.input_pattern.append(
        FLAGS.input_pattern)
    tf.logging.info("Override model input_pattern: %s", FLAGS.input_pattern)

  tf.logging.info("Pipeline configure: %s", '=' * 128)
  tf.logging.info(pipeline_proto)

  topic_list, topic_data = _load_topics(FLAGS.topic_json_input_path,
                                        FLAGS.topic_namelist_txt_input_path)

  if FLAGS.run_once:

    results = _run_prediction(pipeline_proto, topic_list, topic_data)
    tf.logging.info('\n%s', json.dumps(results, indent=2))

  else:
    previous_ckpt = None
    while True:
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
      if checkpoint_path != previous_ckpt:
        global_step = int(checkpoint_path.split('-')[-1])
        if global_step >= FLAGS.eval_min_steps:
          previous_ckpt = checkpoint_path
          metrics = _run_prediction(pipeline_proto, topic_list, topic_data,
                                    checkpoint_path)

          # Write summary.

          summary = tf.Summary()
          for k, v in metrics.items():
            summary.value.add(tag=k, simple_value=v)

          if FLAGS.saved_ckpts_dir:
            best_step, best_metric = save_model_if_it_is_better(
                global_step, metrics['accuracy/micro'], checkpoint_path,
                FLAGS.saved_ckpts_dir)
            summary.value.add(
                tag='accuracy/best', simple_value=best_metric)

          summary_writer = tf.summary.FileWriter(FLAGS.eval_log_dir)
          summary_writer.add_summary(summary, global_step=global_step)
          summary_writer.close()

          if global_step >= FLAGS.max_steps:
            break

          continue

      tf.logging.info('Sleep for 10 secs.')
      time.sleep(10)

  tf.logging.info('Done')


if __name__ == '__main__':
  tf.app.run()
