from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from core import utils
import ads_reader

from google.protobuf import text_format
from protos import reader_pb2


class AdsReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      input_pattern: "data/ads_test.record-00000-of-00010"
      interleave_cycle_length: 1
      is_training: true
      shuffle_buffer_size: 10
      map_num_parallel_calls: 5
      prefetch_buffer_size: 8000
      batch_size: 1
      max_slogan_num: 10
      max_slogan_len: 10
      knowledge_base_config {
        query_to_id_file: "data/sparql_query2id.txt"
        id_to_comment_file: "data/sparql_id2comment.txt"
        max_tokens_to_keep: 100
        max_comments_per_image: 15
        remove_query: false
      }
    """
    options = reader_pb2.AdsReader()
    text_format.Merge(options_str, options)

    iterator = ads_reader.get_input_fn(options)().make_initializable_iterator()
    inputs = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      sess.run(tf.tables_initializer())
      for _ in range(1):
        values = sess.run(inputs)
        print(values['proposal_num'])
        print(values['proposal_box'])
        print(values['proposal_text_string'])
        print(values['proposal_text_length'])
        print(values['proposal_feature'])
        print(values['slogan_num'])
        print(values['slogan_box'])
        print(values['slogan_text_string'])
        print(values['slogan_text_length'])
        print(values['groundtruth_num'])
        print(values['groundtruth_text_string'])
        print(values['groundtruth_text_length'])
        print(values['question_num'])
        print(values['question_text_string'])
        print(values['question_text_length'])
        print(values['image_id'])
        print(values['slogan_kb_num'])
        print(values['slogan_kb_ids'])
        print(values['slogan_kb_text_string'])
        print(values['slogan_kb_text_length'])
        print(values['slogan_kb_mask'])
        print(values['proposal_label_num'])
        print(values['proposal_label_text'])
        print(values['proposal_label_mask'])

  # def test_create_hash_table(self):
  #   tf.reset_default_graph()

  #   table = tf.contrib.lookup.HashTable(
  #       tf.contrib.lookup.TextFileInitializer(
  #           'output/sparql_query2id.txt',
  #           key_dtype=tf.string,
  #           key_index=0,
  #           value_dtype=tf.string,
  #           value_index=1,
  #           delimiter='\t'),
  #       default_value='')

  #   key = tf.placeholder(shape=[], dtype=tf.string)
  #   value = table.lookup(key)

  #   with self.test_session() as sess:
  #     sess.run(tf.tables_initializer())
  #     # self.assertEqual(
  #     #     sess.run(value, feed_dict={key: b'nike'}),
  #     #     b'http://dbpedia.org/resource/Nike,_Inc.')
  #     # self.assertEqual(
  #     #     sess.run(value, feed_dict={key: b'wwe'}),
  #     #     b'http://dbpedia.org/resource/WWE')
  #     # self.assertEqual(
  #     #     sess.run(value, feed_dict={key: b'diesel'}),
  #     #     b'http://dbpedia.org/resource/Diesel_(brand) http://dbpedia.org/resource/Diesel_(band)')

  # def test_process_batch_query(self):
  #   tf.reset_default_graph()

  #   kb_dict = ads_reader.KnowledgeBaseDict('sparql_query2id.txt',
  #                                               'sparql_id2comment.txt')

  #   # sentence shape = [batch, max_sent_num, max_sent_len]

  #   sentence = tf.placeholder(shape=[2, None, None], dtype=tf.string)
  #   results = kb_dict.process_batch_query(sentence)

  #   with self.test_session() as sess:
  #     sess.run(tf.tables_initializer())
  #     values = sess.run(
  #         results,
  #         feed_dict={
  #             sentence: [[['nike', 'is', 'a', 'company', '', ''],
  #                         ['diesel', 'is', 'a', 'brand', 'for', 'honda']],
  #                        [['nike', '', '', '', '', ''],
  #                         ['honda', '', '', '', '', '']]]
  #         })
  #     print(values)


if __name__ == '__main__':
  tf.test.main()
