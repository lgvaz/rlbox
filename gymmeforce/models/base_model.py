import os
import numpy as np
import tensorflow as tf
from gymmeforce.common.utils import huber_loss
from gymmeforce.models.q_graphs import deepmind_graph, simple_graph


# TODO: Global steps
class BaseModel:
    def __init__(self, state_shape, num_actions, input_type=None, log_dir=None):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.global_step_tensor = tf.Variable(1, name='global_step', trainable=False)
        self.log_dir = log_dir
        self.merged = None
        self._saver = None
        self._writer = None

        # If input is an image defaults to uint8, else defaults to float32
        if input_type is None:
            if len(state_shape) == 3:
                input_type = tf.uint8
            else:
                input_type = tf.float32

        # Model inputs
        self.states_t_ph = tf.placeholder(
            name='states',
            shape=[None] + list(self.state_shape),
            dtype=input_type
        )
        self.states_tp1_ph = tf.placeholder(
            name='states_tp1',
            shape=[None] + list(self.state_shape),
            dtype=input_type
        )
        self.actions_ph = tf.placeholder(
            name='actions',
            shape=[None],
            dtype=tf.int32
        )
        self.rewards_ph = tf.placeholder(
            name='rewards',
            shape=[None],
            dtype=tf.float32
        )
        self.dones_ph = tf.placeholder(
            name='done_mask',
            shape=[None],
            dtype=tf.float32
        )
        self.learning_rate_ph = tf.placeholder(
            name='learning_rate_ph',
            shape=[],
            dtype=tf.float32
        )
        self.global_step_ph = tf.placeholder(
            name='global_step_ph',
            shape=[],
            dtype=tf.int32
        )

        if input_type == tf.uint8:
            # Convert to float on GPU
            self.states_t = tf.cast(self.states_t_ph, tf.float32) / 255.
            self.states_tp1 = tf.cast(self.states_tp1_ph, tf.float32) / 255.
        else:
            self.states_t = self.states_t_ph
            self.states_tp1 = self.states_tp1_ph

        self.set_global_step_op = tf.assign(self.global_step_tensor,
                                            self.global_step_ph,
                                            name='step_global_step')
        self.increase_global_step_op = tf.assign_add(self.global_step_tensor, 1,
                                                     name='increase_global_step')

    def _maybe_create_writer(self, logdir):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    def _maybe_create_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver()

    def save(self, sess, step, name='model'):
        self._maybe_create_saver()
        save_path = os.path.join(self.log_dir, name)
        self._saver.save(sess, save_path, global_step=step)

    def load_or_initialize(self, sess, save_path=None):
        ''' Load from checkpoint if exists, else initialize variables '''
        self._maybe_create_saver()
        if save_path is None:
            save_path = tf.train.latest_checkpoint(self.log_dir)
        if save_path is None:
            print('Initializing variables')
            sess.run(tf.global_variables_initializer())
        else:
            print('Loading model from {}'.format(save_path))
            self._saver.restore(sess, save_path)

    def train(self, sess, learning_rate, states_t, states_tp1, actions, rewards, dones):
        feed_dict = {
            self.learning_rate_ph: learning_rate,
            self.states_t_ph: states_t,
            self.states_tp1_ph: states_tp1,
            self.actions_ph: actions,
            self.rewards_ph: rewards,
            self.dones_ph: dones
        }
        sess.run(self.training_op, feed_dict=feed_dict)

    def summary_scalar(self, sess, step, name, value):
        self._maybe_create_writer(self.log_dir)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=value),
        ])
        self._writer.add_summary(summary, step)

    def increase_global_step(self, sess):
        # Increasing the global step every timestep was consuming too much time
        sess.run(self.increase_global_step_op)

    def set_global_step(self, sess, step):
        sess.run(self.set_global_step_op, feed_dict={self.global_step_ph: step})

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
