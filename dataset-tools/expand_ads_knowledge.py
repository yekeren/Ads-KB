from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import nltk
import collections
import numpy as np
import tensorflow as tf

from SPARQLWrapper import SPARQLWrapper, JSON

flags = tf.app.flags

flags.DEFINE_string('ocr_detection_dir', '', 'Path to the OCR detection result directory.')
flags.DEFINE_string('dbpedia_dir', '', 'Path to the output DBpedia directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def _create_corpus(ocr_detection_dir):
  """Creates the slogan corpus from OCR detections.

  Args:
    ocr_detection_dir: Path to the OCR detected slogans.

  Returns:
    A list of strings denoting the texts in the corpus.
  """
  corpus = []
  for filename in os.listdir(ocr_detection_dir):
    filename = os.path.join(ocr_detection_dir, filename)
    with open(filename, 'r', encoding='utf-8') as f:
      data = json.load(f)
    corpus.append(data['text'].replace('\n', ' '))
    if len(corpus) % 1000 == 0:
      tf.logging.info('On %i', len(corpus))
  tf.logging.info("total: %i", len(corpus))
  return corpus


def _create_queries_list(ocr_detection_dir):
  """Creates queries list."""
  corpus = _create_corpus(ocr_detection_dir)
  corpus_tokenized = [nltk.word_tokenize(document) for document in corpus]

  counter = collections.Counter()
  for document in corpus_tokenized:
    for token in document:
      counter[token] += 1
  tf.logging.info('Gathered %i tokens', len(counter))

  stop_words = set(nltk.corpus.stopwords.words('english'))
  queries_list = []
  for word, freq in counter.most_common():
    try:
      word.encode('ascii')
    except Exception as ex:
      continue
    if not word in stop_words:
      queries_list.append(word)
  tf.logging.info('Retained %i tokens', len(queries_list))
  return queries_list


def _is_valid_query(query):
  """Returns true if the query is valid."""
  return 'A' <= query[0] <= 'Z'


def _is_valid_response(response):
  """Returns true if the response is valid."""
  return len(response['results']['bindings']) > 0


def _sparql_query(keyword):
  query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT DISTINCT ?entry ?comment 
    WHERE {
        {
          ?entry rdf:type ?type.
          ?entry rdfs:label "%s"@en.
          ?entry rdfs:comment ?comment.
          } 
        UNION
        {
          ?entry rdf:type ?type.
          ?entry dbpedia2:abbreviation ?abbreviation.
          ?entry rdfs:comment ?comment.
          FILTER regex(?abbreviation, "^%s[^a-zA-Z].*?")
          } 
        UNION
        {
          ?entry rdf:type ?type.
          ?entry_dup rdfs:label "%s"@en.
          ?entry rdfs:comment ?comment.
          ?entry ^dbo:wikiPageDisambiguates|^dbo:wikiPageRedirects ?entry_dup.
          }
        FILTER langMatches(lang(?comment),'en').
        FILTER (?type IN (dbo:Company, dbo:Organisation, dbo:Person, dbo:Software, dbo:VideoGame, dbo:Food, dbo:Weapon, dbo:Place, dbo:Country, dbo:Location, dbo:Beverage, dbo:Agent, dbo:Bank, dbo:EducationalInstitution, dbo:University, dbo:Actor, dbo:Publisher, dbo:Department, dbo:School)) 
        }
    LIMIT 20
  """ % (keyword, keyword, keyword)
  
  sparql = SPARQLWrapper("http://dbpedia.org/sparql")
  sparql.setQuery(query)
  sparql.setReturnFormat(JSON)
  return sparql.query().convert()


def main(_):
  queries_list = _create_queries_list(FLAGS.ocr_detection_dir)

  for i, query in enumerate(queries_list):
    if (i + 1) % 50 == 0:
      tf.logging.info('On sparql %i/%i', i + 1, len(queries_list))

    # Check the validity of the query.

    if not _is_valid_query(query):
      continue

    # Check if we have sent the query.

    filename = os.path.join(FLAGS.dbpedia_dir, '%s.json' % query)
    if os.path.isfile(filename):
      continue
    try:
      response = _sparql_query(keyword=query)
      if _is_valid_response(response):
        tf.logging.info('Saving response for %s', query)
        with open(filename, 'w') as f:
          f.write(json.dumps(response, indent=2))
    except Exception as ex:
      tf.logging.warn(ex)

  tf.logging.info('Done')

if __name__ == '__main__':
  tf.app.run()
