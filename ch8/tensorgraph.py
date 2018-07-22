"""TensorGraph OOP Framework."""

import numpy as np
import tensorflow as tf
import copy
import multiprocessing
import os
import re
import threading
from collections import Sequence

import pickle
import threading
import time

import numpy as np
import os
import six
import tensorflow as tf
import tempfile

class TensorGraph(object):

  def __init__(self,
               batch_size=100,
               random_seed=None,
               graph=None,
               learning_rate=0.001,
               model_dir=None,
               **kwargs):
    """
    Parameters
    ----------
    batch_size: int
      default batch size for training and evaluating
    graph: tensorflow.Graph
      the Graph in which to create Tensorflow objects.  If None, a new Graph
      is created.
    learning_rate: float or LearningRateSchedule
      the learning rate to use for optimization
    kwargs
    """

    # Layer Management
    self.layers = dict()
    self.features = list()
    self.labels = list()
    self.outputs = list()
    self.task_weights = list()
    self.loss = None
    self.built = False
    self.optimizer = None
    self.learning_rate = learning_rate

    # Singular place to hold Tensor objects which don't serialize
    # See TensorGraph._get_tf() for more details on lazy construction
    self.tensor_objects = {
        "Graph": graph,
        #"train_op": None,
    }
    self.global_step = 0

    self.batch_size = batch_size
    self.random_seed = random_seed
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self.model_dir_is_temp = True
    self.model_dir = model_dir
    self.save_file = "%s/%s" % (self.model_dir, "model")
    self.model_class = None

  def _add_layer(self, layer):
    if layer.name is None:
      layer.name = "%s_%s" % (layer.__class__.__name__, len(self.layers) + 1)
    if layer.name in self.layers:
      return
    if isinstance(layer, Input):
      self.features.append(layer)
    self.layers[layer.name] = layer
    for in_layer in layer.in_layers:
      self._add_layer(in_layer)

  def topsort(self):

    def add_layers_to_list(layer, sorted_layers):
      if layer in sorted_layers:
        return
      for in_layer in layer.in_layers:
        add_layers_to_list(in_layer, sorted_layers)
      sorted_layers.append(layer)

    sorted_layers = []
    for l in self.features + self.labels + self.task_weights + self.outputs:
      add_layers_to_list(l, sorted_layers)
    add_layers_to_list(self.loss, sorted_layers)
    return sorted_layers

  def build(self):
    if self.built:
      return
    with self._get_tf("Graph").as_default():
      self._training_placeholder = tf.placeholder(dtype=tf.float32, shape=())
      if self.random_seed is not None:
        tf.set_random_seed(self.random_seed)
      for layer in self.topsort():
        with tf.name_scope(layer.name):
          layer.create_tensor(training=self._training_placeholder)
      self.session = tf.Session()

      self.built = True

  def set_loss(self, layer):
    self._add_layer(layer)
    self.loss = layer

  def add_output(self, layer):
    self._add_layer(layer)
    self.outputs.append(layer)

  def set_optimizer(self, optimizer):
    """Set the optimizer to use for fitting."""
    self.optimizer = optimizer

  def get_layer_variables(self, layer):
    """Get the list of trainable variables in a layer of the graph."""
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      if layer.variable_scope == "":
        return []
      return tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=layer.variable_scope)

  def get_global_step(self):
    return self._get_tf("GlobalStep")

  def _get_tf(self, obj):
    """Fetches underlying TensorFlow primitives.

    Parameters
    ----------
    obj: str
      If "Graph", returns tf.Graph instance. If "Optimizer", returns the
      optimizer. If "train_op", returns the train operation. If "GlobalStep" returns
      the global step.
    Returns
    -------
    TensorFlow Object

    """

    if obj in self.tensor_objects and self.tensor_objects[obj] is not None:
      return self.tensor_objects[obj]
    if obj == "Graph":
      self.tensor_objects["Graph"] = tf.Graph()
    elif obj == "Optimizer":
      self.tensor_objects["Optimizer"] = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-7)
    elif obj == "GlobalStep":
      with self._get_tf("Graph").as_default():
        self.tensor_objects["GlobalStep"] = tf.Variable(0, trainable=False)
    return self._get_tf(obj)

  def restore(self):
    """Reload the values of all variables from the most recent checkpoint file."""
    if not self.built:
      self.build()
    last_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if last_checkpoint is None:
      raise ValueError("No checkpoint found")
    with self._get_tf("Graph").as_default():
      saver = tf.train.Saver()
      saver.restore(self.session, last_checkpoint)

  def __del__(self):
    pass

class Layer(object):

  def __init__(self, in_layers=None, **kwargs):
    if "name" in kwargs:
      self.name = kwargs["name"]
    else:
      self.name = None
    if in_layers is None:
      in_layers = list()
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    self.in_layers = in_layers
    self.variable_scope = ""
    self.tb_input = None

  def create_tensor(self, in_layers=None, **kwargs):
    raise NotImplementedError("Subclasses must implement for themselves")

  def _get_input_tensors(self, in_layers):
    """Get the input tensors to his layer.

    Parameters
    ----------
    in_layers: list of Layers or tensors
      the inputs passed to create_tensor().  If None, this layer's inputs will
      be used instead.
    """
    if in_layers is None:
      in_layers = self.in_layers
    if not isinstance(in_layers, Sequence):
      in_layers = [in_layers]
    tensors = []
    for input in in_layers:
      tensors.append(tf.convert_to_tensor(input))
    return tensors

def _convert_layer_to_tensor(value, dtype=None, name=None, as_ref=False):
  return tf.convert_to_tensor(value.out_tensor, dtype=dtype, name=name)


tf.register_tensor_conversion_function(Layer, _convert_layer_to_tensor)

class Dense(Layer):

  def __init__(
      self,
      out_channels,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer,
      weights_initializer=None,
      biases_regularizer=None,
      weights_regularizer=None,
      trainable=True,
      **kwargs):
    """Create a dense layer.

    The weight and bias initializers are specified by callable objects that construct
    and return a Tensorflow initializer when invoked with no arguments.  This will typically
    be either the initializer class itself (if the constructor does not require arguments),
    or a TFWrapper (if it does).

    Parameters
    ----------
    out_channels: int
      the number of output values
    activation_fn: object
      the Tensorflow activation function to apply to the output
    biases_initializer: callable object
      the initializer for bias values.  This may be None, in which case the layer
      will not include biases.
    weights_initializer: callable object
      the initializer for weight values
    """
    super(Dense, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.out_tensor = None
    self.activation_fn = activation_fn
    self.biases_initializer = biases_initializer
    self.weights_initializer = weights_initializer
    self.biases_regularizer = biases_regularizer
    self.weights_regularizer = weights_regularizer
    self.trainable = trainable

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Dense layer can only have one input")
    parent = inputs[0]
    if self.biases_initializer is None:
      biases_initializer = None
    else:
      biases_initializer = self.biases_initializer

    if self.weights_initializer is None:
      weights_initializer = tf.contrib.layers.variance_scaling_initializer()
    else:
      weights_initializer = self.weights_initializer

    out_tensor = tf.contrib.layers.fully_connected(parent,
                                                   num_outputs=self.out_channels,
                                                   activation_fn=self.activation_fn,
                                                   biases_initializer=biases_initializer,
                                                   weights_initializer=weights_initializer,
                                                   biases_regularizer=self.biases_regularizer,
                                                   weights_regularizer=self.weights_regularizer,
                                                   reuse=False,
                                                   trainable=self.trainable)
    self.out_tensor = out_tensor
    return out_tensor

class Squeeze(Layer):

  def __init__(self, in_layers=None, squeeze_dims=None, **kwargs):
    self.squeeze_dims = squeeze_dims
    super(Squeeze, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.squeeze(parent_tensor, squeeze_dims=self.squeeze_dims)
    self.out_tensor = out_tensor
    return out_tensor

class BatchNorm(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(BatchNorm, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    parent_tensor = inputs[0]
    out_tensor = tf.layers.batch_normalization(parent_tensor)
    self.out_tensor = out_tensor
    return out_tensor

class Flatten(Layer):
  """Flatten every dimension except the first"""

  def __init__(self, in_layers=None, **kwargs):
    super(Flatten, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Only One Parent to Flatten")
    parent = inputs[0]
    parent_shape = parent.get_shape()
    vector_size = 1
    for i in range(1, len(parent_shape)):
      vector_size *= parent_shape[i].value
    parent_tensor = parent
    out_tensor = tf.reshape(parent_tensor, shape=(-1, vector_size))
    self.out_tensor = out_tensor
    return out_tensor

class SoftMax(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(SoftMax, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    if len(inputs) != 1:
      raise ValueError("Must only Softmax single parent")
    parent = inputs[0]
    out_tensor = tf.contrib.layers.softmax(parent)
    self.out_tensor = out_tensor
    return out_tensor

class Input(Layer):

  def __init__(self, shape, dtype=tf.float32, **kwargs):
    self._shape = tuple(shape)
    self.dtype = dtype
    super(Input, self).__init__(**kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    if in_layers is None:
      in_layers = self.in_layers
    out_tensor = tf.placeholder(dtype=self.dtype, shape=self._shape)
    self.out_tensor = out_tensor
    return out_tensor

class Add(Layer):

  def __init__(self, in_layers=None, constants=None, bias=None, **kwargs):
    self.constants = constants
    self.bias = bias
    super(Add, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    out_tensor = self.constants[0] * inputs[0]
    for i in range(1, len(inputs)):
      out_tensor = out_tensor + self.constants[i] * inputs[i]

    if self.bias != None:
      out_tensor = out_tensor + self.bias

    self.out_tensor = out_tensor
    return out_tensor

class Scale(Layer):

  def __init__(self, in_layers=None, initial_value=1.0, trainable=True, **kwargs):
    self.initial_value=initial_value
    self.trainable=trainable
    super(Scale, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)

    scale_weight = tf.Variable(initial_value=self.initial_value,name="scale_weight",trainable=self.trainable)

    out_tensor = scale_weight * inputs[0]
    self.out_tensor = out_tensor
    return out_tensor


class MaxValue(Layer):

  def __init__(self, in_layers=None, **kwargs):
    super(MaxValue, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    input0 = inputs[0] # shape = (Nplays, 9)
    indices = tf.argmax(input0, axis=1) # [...,tf.newaxis] # ==> [[i0][i0]..[iNplays-1]]
    shape0 = tf.shape(indices)[0]   # ==> Nplays
    shape_range = tf.range(shape0) # [...,tf.newaxis] # ==> [[0][1]...[Nplays-1]]
    gather_indices = tf.stack([tf.to_int64(shape_range), indices], axis=1) # ==> [[0, i0][1, i1]..[Nplays-1, iNplays-1]]
    out_tensor = tf.gather_nd(input0, gather_indices) # apply simple indexing into matrix input0
    self.out_tensor = out_tensor
    return out_tensor


class Conv2D(Layer):
  def __init__(self, name, shape, in_layers=None, **kwargs):
    super(Conv2D, self).__init__(in_layers, **kwargs)
    self.name = name
    self.shape = shape

  def create_tensor(self, in_layers=None, **kwargs):
    with tf.variable_scope(self.name):
      conv_weights = tf.Variable(
          tf.truncated_normal(self.shape,
                          stddev=0.1,
                          seed=None, dtype=tf.float32))
      conv_biases = tf.Variable(tf.zeros([self.shape[3]], dtype=tf.float32))

      inputs = self._get_input_tensors(in_layers)
      input0 = inputs[0]
      conv = tf.nn.conv2d(input0,
                      conv_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

      # Bias and rectified linear non-linearity.

      relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

      # Max pooling. The kernel size spec {ksize} also follows the layout
      # of the data. Here we have a pooling window of 2, and a stride of
      # 2.
      pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

      out_tensor = pool
      self.out_tensor = out_tensor
      return out_tensor


class MultiRNN(Layer):
  def __init__(self, in_layers=None, is_training=False, keep_prob=1.0, cell_size=84, num_layers=3, **kwargs):
    self.is_training = is_training
    self.keep_prob = keep_prob
    self.cell_size = cell_size
    self.num_layers = num_layers
    super(MultiRNN, self).__init__(in_layers, **kwargs)

  def create_tensor(self, in_layers=None, **kwargs):
    inputs = self._get_input_tensors(in_layers)
    input0 = inputs[0]

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=0.0, state_is_tuple=True,
                                             reuse=tf.get_variable_scope().reuse)

    attn_cell = lstm_cell
    if self.is_training and self.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)

    attn_cells = [attn_cell() for _ in range(self.num_layers)]
    out_tensor = tf.contrib.rnn.MultiRNNCell(attn_cells, state_is_tuple=True)

  #  TODO
  #   state = self.initial_state
  # with tf.variable_scope("RNN"):
  #   for time_step in range(num_steps):
  #     if time_step > 0: tf.get_variable_scope().reuse_variables()
  #     (cell_output, state) = cell(inputs[:, time_step, :], state)
  #     outputs.append(cell_output)

    self.out_tensor = out_tensor
    return out_tensor
