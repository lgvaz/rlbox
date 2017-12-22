import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from gymmeforce.models.q_graphs import deepmind_graph, simple_graph


class BaseModel:
    def __init__(self, env_config, grad_clip_norm=None, log_dir='logs/examples',
                 **kwargs):
        self.env_config = env_config
        self.grad_clip_norm = grad_clip_norm
        self.log_dir = log_dir
        self.placeholders = {}
        self.training_op = None
        self.merged = None
        self._saver = None
        self._writer = None
        self.callbacks = []

        self.global_step_sy = tf.Variable(1, name='global_step', trainable=False)
        placeholders_config = {'add_to_global_step': [[], tf.int32]}
        self._create_placeholders(placeholders_config)
        self.increase_global_step_op = tf.assign_add(
            self.global_step_sy,
            self.placeholders['add_to_global_step'],
            name='increase_global_step')

    def _create_placeholders(self, config):
        for name, (shape, dtype) in config.items():
            self.placeholders[name] = tf.placeholder(dtype, shape, name)

    def _maybe_create_writer(self):
        if self._writer is None:
            print('Writing logs to: {}'.format(self.log_dir))
            self._writer = tf.summary.FileWriter(
                self.log_dir, graph=tf.get_default_graph())

    def _maybe_create_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver()

    def _create_training_op(self,
                            learning_rate,
                            opt=tf.train.AdamOptimizer,
                            opt_config=dict(),
                            var_list=None):
        loss_sy = tf.losses.get_total_loss()
        optimizer = opt(learning_rate, **opt_config)
        grads_and_vars = optimizer.compute_gradients(loss_sy, var_list=var_list)

        if self.grad_clip_norm is not None:
            with tf.variable_scope('gradient_clipping'):
                grads_and_vars = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                  for grad, var in grads_and_vars if grad is not None]

        self.training_op = optimizer.apply_gradients(grads_and_vars)

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

    def _fetch_placeholders_data_dict(self, batch):
        '''
        Create a dictionary mapping placeholders to their correspondent value
        Modify this method to include new placeholders to feed_dict used by training_op
        '''
        self.placeholders_and_data = {
            self.placeholders[key]: value
            for key, value in batch.items() if key in self.placeholders.keys()
        }

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
        sess.run(
            self.increase_global_step_op,
            feed_dict={
                self.placeholders['add_to_global_step']: value
            })

    def get_global_step(self, sess):
        return tf.train.global_step(sess, self.global_step_sy)

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
