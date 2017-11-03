import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from gymmeforce.models.q_graphs import deepmind_graph, simple_graph


# TODO: Global steps
class BaseModel:
    def __init__(self, env_config, log_dir='logs/examples', **kwargs):
        self.env_config = env_config
        self.log_dir = log_dir
        self.placeholders = {}
        self.training_op = None
        self.merged = None
        self._saver = None
        self._writer = None

        self.global_step_sy = tf.Variable(1, name='global_step', trainable=False)
        placeholders_config = {
            'add_to_global_step': [[], tf.int32]
        }
        self._create_placeholders(placeholders_config)
        self.increase_global_step_op = tf.assign_add(self.global_step_sy,
                                                     self.placeholders['add_to_global_step'],
                                                     name='increase_global_step')

    def _create_placeholders(self, config):
        for name, (shape, dtype) in config.items():
            self.placeholders[name] = tf.placeholder(dtype, shape, name)

    def _maybe_create_writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

    def _maybe_create_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver()

    def _create_training_op(self, learning_rate, opt=tf.train.AdamOptimizer):
        # TODO: Change to do grad clipping and stuff
        loss_sy = tf.losses.get_total_loss()
        self.training_op = opt(learning_rate, epsilon=1e-5).minimize(loss_sy)

    def _create_summaries_op(self):
        self._maybe_create_writer()
        # Add all losses to the summary
        losses = tf.losses.get_losses()
        for tensor in losses:
            name = '/'.join(['losses'] + tensor.name.split('/')[:-1])
            tf.summary.scalar(name, tensor)

    def _write_summaries(self, sess, feed_dict):
        if self.merged is None:
            self._create_summaries_op()
            self.merged = tf.summary.merge_all()

        summary = sess.run(self.merged, feed_dict=feed_dict)
        self._writer.add_summary(summary, self.get_global_step(sess))

    def save(self, sess, name='model'):
        self._maybe_create_saver()
        save_path = os.path.join(self.log_dir, name)
        self._saver.save(sess, save_path, global_step=self.global_step_sy)

    def load_or_initialize(self, sess, save_path=None):
        ''' Load from checkpoint if exists, else initialize variables '''
        self._maybe_create_saver()
        if save_path is None:
            save_path = tf.train.latest_checkpoint(self.log_dir)
        if save_path is None:
            print('Initializing variables')
            self.initialize(sess)
        else:
            print('Loading model from {}'.format(save_path))
            self._saver.restore(sess, save_path)

    def summary_scalar(self, sess, name, value, step=None):
        if step is None:
            step = self.get_global_step(sess)
        self._maybe_create_writer()
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=value),
        ])
        self._writer.add_summary(summary, step)

    def increase_global_step(self, sess, value):
        sess.run(self.increase_global_step_op,
                 feed_dict={self.placeholders['add_to_global_step']: value})

    def get_global_step(self, sess):
        return tf.train.global_step(sess, self.global_step_sy)

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
