import os
import numpy as np
import tensorflow as tf
from gymmeforce.models.q_graphs import deepmind_graph, simple_graph


# TODO: Global steps
class BaseModel:
    def __init__(self, env_config, log_dir=None):
        self.env_config = env_config
        self.log_dir = log_dir
        self.training_op = None
        self.merged = None
        self._saver = None
        self._writer = None
        self.placeholders = dict()
        self.global_step_tensor = tf.Variable(1, name='global_step', trainable=False)
        self.increase_global_step_op = tf.assign_add(self.global_step_tensor, 1,
                                                     name='increase_global_step')

    def _create_placeholders(self, config):
        for name, (shape, dtype) in config.items():
            self.placeholders[name] = tf.placeholder(dtype, shape, name)

    def _maybe_create_writer(self, logdir):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    def _maybe_create_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver()

    def _create_training_op(self, learning_rate, opt=tf.train.AdamOptimizer):
        # TODO: Change to do grad clipping and stuff
        loss = tf.losses.get_total_loss()
        self.training_op = opt(learning_rate).minimize(loss)

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
