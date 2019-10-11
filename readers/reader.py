from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import reader_pb2
from readers import wsod_reader
from readers import advise_reader
from readers import ads_reader


def get_input_fn(options):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.Reader):
    raise ValueError('options has to be an instance of Reader.')

  reader_oneof = options.WhichOneof('reader_oneof')

  if 'ads_reader' == reader_oneof:
    return ads_reader.get_input_fn(options.ads_reader)

  raise ValueError('Invalid reader %s' % (reader_oneof))
