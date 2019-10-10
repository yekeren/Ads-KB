from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import collections
import nltk

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('dbpedia_dir', '',
                    'Path to the directory saving query expansion data.')
flags.DEFINE_string('query_to_id_file', '', 'Path to the output file.')
flags.DEFINE_string('id_to_comment_file', '', 'Path to the output file.')
flags.DEFINE_integer('top_k_entries', 10,
                     'Use only the top-k to build knowledge base.')

FLAGS = flags.FLAGS


def _load_dbpedia_comment(path, top_k_entries):
  """Loads DBPedia comments."""
  comments = collections.defaultdict(list)
  entries = set()
  for filename in os.listdir(path):
    query = filename.split('.json')[0]
    filename = os.path.join(path, filename)
    with open(filename, 'r') as f:
      data = json.load(f)
    bindings = data['results']['bindings']
    for binding in bindings[:top_k_entries]:
      entry = binding['entry']['value']
      entries.add(entry)
      comment = binding['comment']['value']
      try:
        comments[query].append((entry, comment))
      except Exception as ex:
        pass
  tf.logging.info('Load %i anchored queries.', len(comments))
  tf.logging.info('Load %i sparql KB entries.', len(entries))
  return comments


def _tokenize(sentence):
  """Seperates the sentence into tokens.  """
  tokens = nltk.word_tokenize(sentence.lower())
  return tokens


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  query_to_comments = _load_dbpedia_comment(FLAGS.dbpedia_dir,
                                            FLAGS.top_k_entries)
  query_to_id = {}
  id_to_comment = {}

  for query, data in query_to_comments.items():
    for url, comment in data:
      try:
        url.encode('ascii')
        id_to_comment[url] = comment
        query_to_id.setdefault(query.lower(), set([])).add(url)
      except Exception as ex:
        pass

  with open(FLAGS.id_to_comment_file, 'w', encoding="utf8") as f:
    for url, comment in id_to_comment.items():
      comment = _tokenize(comment)
      comment = ' '.join(comment)
      f.write('%s\t%s\n' % (url, comment))

  with open(FLAGS.query_to_id_file, 'w', encoding='utf8') as f:
    for query, ids in query_to_id.items():
      f.write('%s\t%s\n' % (query, ' '.join(ids)))


if __name__ == '__main__':
  tf.app.run()
