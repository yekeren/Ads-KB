from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from protos import hyperparams_pb2
from protos import optimizer_pb2
from object_detection.utils import learning_schedules

slim = tf.contrib.slim


def create_optimizer(options, learning_rate=0.1):
  """Builds optimizer from options.

  Args:
    options: an instance of optimizer_pb2.Optimizer.
    learning_rate: a scalar tensor denoting the learning rate.

  Returns:
    a tensorflow optimizer instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, optimizer_pb2.Optimizer):
    raise ValueError('The options has to be an instance of Optimizer.')

  optimizer = options.WhichOneof('optimizer')

  if 'sgd' == optimizer:
    options = options.sgd
    return tf.train.GradientDescentOptimizer(
        learning_rate, use_locking=options.use_locking)

  if 'momentum' == optimizer:
    options = options.momentum
    return tf.train.MomentumOptimizer(
        learning_rate,
        momentum=options.momentum,
        use_locking=options.use_locking,
        use_nesterov=options.use_nesterov)

  if 'adagrad' == optimizer:
    options = options.adagrad
    return tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=options.initial_accumulator_value,
        use_locking=options.use_locking)

  if 'adam' == optimizer:
    options = options.adam
    return tf.train.AdamOptimizer(
        learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        use_locking=options.use_locking)

  if 'rmsprop' == optimizer:
    options = options.rmsprop
    return tf.train.RMSPropOptimizer(
        learning_rate,
        decay=options.decay,
        momentum=options.momentum,
        epsilon=options.epsilon,
        use_locking=options.use_locking,
        centered=options.centered)

  raise ValueError('Invalid optimizer: {}.'.format(optimizer))


def create_learning_rate(learning_rate_config):
  """Create optimizer learning rate based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  learning_rate = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
  if learning_rate_type == 'constant_learning_rate':
    config = learning_rate_config.constant_learning_rate
    learning_rate = tf.constant(config.learning_rate, dtype=tf.float32,
                                name='learning_rate')

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    learning_rate = learning_schedules.exponential_decay_with_burnin(
        tf.train.get_or_create_global_step(),
        config.initial_learning_rate,
        config.decay_steps,
        config.decay_factor,
        burnin_learning_rate=config.burnin_learning_rate,
        burnin_steps=config.burnin_steps,
        min_learning_rate=config.min_learning_rate,
        staircase=config.staircase)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    learning_rate = learning_schedules.manual_stepping(
        tf.train.get_or_create_global_step(), learning_rate_step_boundaries,
        learning_rate_sequence, config.warmup)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    learning_rate = learning_schedules.cosine_decay_with_warmup(
        tf.train.get_or_create_global_step(),
        config.learning_rate_base,
        config.total_steps,
        config.warmup_learning_rate,
        config.warmup_steps,
        config.hold_base_rate_steps)

  if learning_rate is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return learning_rate


class IdentityContextManager(object):
  """Returns an identity context manager that does nothing.

  This is helpful in setting up conditional `with` statement as below:

  with slim.arg_scope(x) if use_slim_scope else IdentityContextManager():
    do_stuff()

  """

  def __enter__(self):
    return None

  def __exit__(self, exec_type, exec_value, traceback):
    del exec_type
    del exec_value
    del traceback
    return False


def build_hyperparams(options, is_training):
  """Builds tf-slim arg_scope for tensorflow ops.

  Args:
    options: an hyperparams_pb2.Hyperparams instance.
    is_training: whether the network is in training mode.

  Returns:
    tf-slim arg_scope containing hyperparameters for ops.

  Raises:
    ValueError: if the options is invalid.
  """
  if not isinstance(options, hyperparams_pb2.Hyperparams):
    raise ValueError('The options has to be an instance of Hyperparams.')

  batch_norm = None
  batch_norm_params = None
  if options.HasField('batch_norm'):
    batch_norm = slim.batch_norm
    batch_norm_params = _build_batch_norm_params(options.batch_norm,
                                                 is_training)

  affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
  if options.op == hyperparams_pb2.Hyperparams.FC:
    affected_ops = [slim.fully_connected]

  with (slim.arg_scope([slim.batch_norm], **batch_norm_params)
        if batch_norm_params is not None else IdentityContextManager()):
    with slim.arg_scope(
        affected_ops,
        weights_regularizer=_build_slim_regularizer(options.regularizer),
        weights_initializer=_build_initializer(options.initializer),
        activation_fn=_build_activation_fn(options.activation),
        normalizer_fn=batch_norm) as sc:
      return sc


def _build_activation_fn(activation_fn):
  """Builds a callable activation from config.

  Args:
    activation_fn: hyperparams_pb2.Hyperparams.activation

  Returns:
    Callable activation function.

  Raises:
    ValueError: On unknown activation function.
  """
  if activation_fn == hyperparams_pb2.Hyperparams.NONE:
    return None
  if activation_fn == hyperparams_pb2.Hyperparams.RELU:
    return tf.nn.relu
  if activation_fn == hyperparams_pb2.Hyperparams.RELU_6:
    return tf.nn.relu6
  raise ValueError('Unknown activation function: {}'.format(activation_fn))


def _build_slim_regularizer(regularizer):
  """Builds a tf-slim regularizer from config.

  Args:
    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf-slim regularizer.

  Raises:
    ValueError: On unknown regularizer.
  """
  regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
  if regularizer_oneof == 'l1_regularizer':
    return slim.l1_regularizer(scale=float(regularizer.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    return slim.l2_regularizer(scale=float(regularizer.l2_regularizer.weight))
  if regularizer_oneof is None:
    return None
  raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))


def _build_initializer(initializer):
  """Build a tf initializer from config.

  Args:
    initializer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf initializer.

  Raises:
    ValueError: On unknown initializer.
  """
  initializer_oneof = initializer.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=initializer.truncated_normal_initializer.mean,
        stddev=initializer.truncated_normal_initializer.stddev)
  if initializer_oneof == 'random_normal_initializer':
    return tf.random_normal_initializer(
        mean=initializer.random_normal_initializer.mean,
        stddev=initializer.random_normal_initializer.stddev)
  if initializer_oneof == 'variance_scaling_initializer':
    enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.DESCRIPTOR.
                       enum_types_by_name['Mode'])
    mode = enum_descriptor.values_by_number[
        initializer.variance_scaling_initializer.mode].name
    return slim.variance_scaling_initializer(
        factor=initializer.variance_scaling_initializer.factor,
        mode=mode,
        uniform=initializer.variance_scaling_initializer.uniform)
  if initializer_oneof == 'glorot_normal_initializer':
    return tf.glorot_normal_initializer()
  if initializer_oneof == 'glorot_uniform_initializer':
    return tf.glorot_uniform_initializer()

  raise ValueError('Unknown initializer function: {}'.format(initializer_oneof))


def _build_batch_norm_params(batch_norm, is_training):
  """Build a dictionary of batch_norm params from config.

  Args:
    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.
    is_training: Whether the models is in training mode.

  Returns:
    A dictionary containing batch_norm parameters.
  """
  batch_norm_params = {
      'decay': batch_norm.decay,
      'center': batch_norm.center,
      'scale': batch_norm.scale,
      'epsilon': batch_norm.epsilon,
      'is_training': is_training and batch_norm.train,
  }
  return batch_norm_params


def get_best_model_checkpoint(saved_ckpts_dir):
  """Gets the path of the best checkpoint.

  Args:
    saved_ckpt_dir: the directory used to save the best model.

  Returns:
    path to the best checkpoint.
  """
  filename = 'saved_info.txt'
  filename = os.path.join(saved_ckpts_dir, filename)
  assert tf.gfile.Exists(filename)

  with open(filename, 'r') as fp:
    step_best, metric_best = fp.readline().strip().split('\t')

  ckpt_path = os.path.join(saved_ckpts_dir, 'model.ckpt-{}'.format(step_best))

  assert tf.gfile.Exists(ckpt_path+ '.meta')
  assert tf.gfile.Exists(ckpt_path+ '.index')
  return ckpt_path


def save_model_if_it_is_better(global_step, model_metric, 
    model_path, saved_ckpts_dir, reverse=False):
  """Saves model if it is better than previous best model.

  The function backups model checkpoint if it is a better model.

  Args:
    global_step: a integer denoting current global step.
    model_metric: a float number denoting performance of current model.
    model_path: current model path.
    saved_ckpt_dir: the directory used to save the best model.
    reverse: if True, smaller value means better model.

  Returns:
    step_best: global step of the best model.
    metric_best: performance of the best model.
  """
  tf.gfile.MakeDirs(saved_ckpts_dir)

  # Read the record file to get the previous best model.
  filename = 'saved_info.txt'
  filename = os.path.join(saved_ckpts_dir, filename)

  step_best, metric_best = None, None
  if tf.gfile.Exists(filename):
    with open(filename, 'r') as fp:
      step_best, metric_best = fp.readline().strip().split('\t')
    step_best, metric_best = int(step_best), float(metric_best)

  condition = lambda x, y: (x > y) if not reverse else (x < y)

  if metric_best is None or condition(model_metric, metric_best):
    tf.logging.info(
        'Current model[%.4lf] is better than the previous best one[%.4lf].',
        model_metric, 0.0 if metric_best is None else metric_best)
    step_best, metric_best = global_step, model_metric

    # Backup checkpoint files.
    tf.logging.info('Copying files...')

    with open(filename, 'w') as fp:
      fp.write('%d\t%.8lf' % (global_step, model_metric))

    for existing_path in tf.gfile.Glob(
        os.path.join(saved_ckpts_dir, 'model.ckpt*')):
      tf.gfile.Remove(existing_path)
      tf.logging.info('Remove %s.', existing_path)

    for source_path in tf.gfile.Glob(model_path + '*'):
      dest_path = os.path.join(saved_ckpts_dir, os.path.split(source_path)[1])
      tf.gfile.Copy(source_path, dest_path, overwrite=True)
      tf.logging.info('Copy %s to %s.', source_path, dest_path)
  return step_best, metric_best
