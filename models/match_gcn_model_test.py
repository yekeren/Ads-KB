from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from core import utils
from models import match_gcn_model


class MatchGcnModelTest(tf.test.TestCase):

  def test_create_hash_table(self):
    tf.reset_default_graph()

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.TextFileInitializer(
            'output/sparql_query2id.txt',
            key_dtype=tf.string,
            key_index=0,
            value_dtype=tf.string,
            value_index=1,
            delimiter='\t'),
        default_value='')

    key = tf.placeholder(shape=[], dtype=tf.string)
    value = table.lookup(key)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      # self.assertEqual(
      #     sess.run(value, feed_dict={key: b'nike'}),
      #     b'http://dbpedia.org/resource/Nike,_Inc.')
      # self.assertEqual(
      #     sess.run(value, feed_dict={key: b'wwe'}),
      #     b'http://dbpedia.org/resource/WWE')
      # self.assertEqual(
      #     sess.run(value, feed_dict={key: b'diesel'}),
      #     b'http://dbpedia.org/resource/Diesel_(brand) http://dbpedia.org/resource/Diesel_(band)')

  def test_process_batch_query(self):
    tf.reset_default_graph()

    kb_dict = match_gcn_model.KnowledgeBaseDict('sparql_query2id.txt',
                                                'sparql_id2comment.txt')

    # sentence shape = [batch, max_sent_num, max_sent_len]

    sentence = tf.placeholder(shape=[2, None, None], dtype=tf.string)
    results = kb_dict.process_batch_query(sentence)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      values = sess.run(
          results,
          feed_dict={
              sentence: [[['nike', 'is', 'a', 'company', '', ''],
                          ['diesel', 'is', 'a', 'brand', 'for', 'honda']],
                         [['nike', '', '', '', '', ''],
                          ['honda', '', '', '', '', '']]]
          })
      print(values)
      print(values[0][0])
      print(values[5][0])
      print(values[0][1])
      print(values[5][1])
      #print(values[0].shape)
      # print(values[2])
      # print(values[0])
      # print(values[3])


if __name__ == '__main__':
  tf.test.main()
