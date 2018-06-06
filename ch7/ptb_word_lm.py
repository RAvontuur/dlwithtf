# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import reader
import tensorflow as tf
import time

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
FLAGS = flags.FLAGS


def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_data, targets):

        size = config.hidden_size
        vocab_size = config.vocab_size
        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell(size):
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(size), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell(size) for _ in range(config.num_layers)],
            state_is_tuple=True)

        self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf.reshape(logits, [config.batch_size, config.num_steps, vocab_size])

        # use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([config.batch_size, config.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # update the cost variables
        self.cost = tf.reduce_sum(loss)
        self.final_state = state


class PTBModelTraining(object):
    def __init__(self, config, data):

        with tf.name_scope("Train"):
            input_data, targets = reader.ptb_producer(data, config.batch_size, config.num_steps, name="TrainInput")

            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_data=input_data, targets=targets)

        self.data_size = len(data)
        self.cost = m.cost
        self.initial_state = m.initial_state
        self.final_state = m.final_state

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self.new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


class PTBModelValidation(object):
    def __init__(self, config, data, name):
        with tf.name_scope(name):
            # epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
            input_data, targets = reader.ptb_producer(data, config.batch_size, config.num_steps, name=name + "Input")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                m = PTBModel(is_training=False, config=config, input_data=input_data, targets=targets)
        self.data_size = len(data)
        self.cost = m.cost
        self.initial_state = m.initial_state
        self.final_state = m.final_state


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, config, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    epoch_size = ((model.data_size // config.batch_size) - 1) // config.num_steps

    for step in range(epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += config.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size,
                   np.exp(costs / iters),
                   (iters
                    * config.batch_size / (time.time() - start_time))))

    return np.exp(costs / iters)


raw_data = reader.ptb_raw_data("./simple-examples/data")
train_data, valid_data, test_data, word_to_id = raw_data

print('fireman = {}'.format(word_to_id['fireman']))
play_data = [word_to_id['fireman']]
print('play_data = {}'.format(play_data))

print('len train_data = {}'.format(len(train_data)))
print('len valid_data = {}'.format(len(valid_data)))
print('len test_data = {}'.format(len(test_data)))
train_data = train_data[0:11000]
print('len traindata = {}'.format(len(train_data)))

with tf.Graph().as_default():
    config = SmallConfig()
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    m = PTBModelTraining(config=config, data=train_data)
    tf.summary.scalar("Training Loss", m.cost)
    tf.summary.scalar("Learning Rate", m.lr)

    mvalid = PTBModelValidation(config=config, data=valid_data, name="Valid")
    tf.summary.scalar("Validation Loss", mvalid.cost)

    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    mtest = PTBModelValidation(config=eval_config, data=test_data, name="Test")

    sv = tf.train.MonitoredTrainingSession()
    with sv as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f"
                  % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, config, eval_op=m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f"
                  % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, config)
            print("Epoch: %d Valid Perplexity: %.3f"
                  % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, eval_config)
        print("Test Perplexity: %.3f" % test_perplexity)
