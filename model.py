import numpy as np
import tensorflow as tf
from utils import huber_loss
from graphs import deepmind_graph, simple_graph
from base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, state_shape, num_actions, graph=None,
                 input_type=None, clip_norm=10, gamma=0.99, double=False, log_dir=None):
        super(DQNModel, self).__init__(state_shape, num_actions, input_type, log_dir)
        self.double = double

        # If input is an image defaults to deepmind_graph, else simple_graph
        if graph is None:
            if len(state_shape) == 3:
                graph = deepmind_graph
                print('Using deepmind_graph')
            else:
                graph = simple_graph
                print('Using simple_graph')
        else:
            print('Using custom graph')

        # Create graphs
        self.q_online_t = graph(self.states_t, num_actions, 'online')
        self.q_target_tp1 = graph(self.states_tp1, num_actions, 'target')
        if double:
            self.q_online_tp1 = graph(self.states_tp1, num_actions,
                                      'online', reuse=True)

        # Create training operation
        self.training_op = self._build_optimization(clip_norm, gamma)
        self.update_target_op = self._build_target_update_op()

        # Create summaries
        if log_dir is not None:
            self.merged = self._create_summaries_op()

        # Create collections for loading later
        tf.add_to_collection('state_input', self.states_t_ph)
        tf.add_to_collection('q_online_t', self.q_online_t)

    def _build_optimization(self, clip_norm, gamma):
        # Choose only the q values for selected actions
        onehot_actions = tf.one_hot(self.actions_ph, self.num_actions)
        q_t = tf.reduce_sum(tf.multiply(self.q_online_t, onehot_actions), axis=1)

        # Caculate td_target
        if self.double:
            best_actions_onehot = tf.one_hot(tf.argmax(self.q_online_tp1, axis=1), self.num_actions)
            q_tp1 = tf.reduce_sum(tf.multiply(self.q_target_tp1, best_actions_onehot), axis=1)
        else:
            q_tp1 = tf.reduce_max(self.q_target_tp1, axis=1)
        td_target = self.rewards_ph + (1 - self.dones_ph) * gamma * q_tp1
        # errors = tf.squared_difference(q_t, td_target)
        errors = huber_loss(q_t, td_target)
        self.total_error = tf.reduce_mean(errors)

        # Create training operation
        opt = tf.train.AdamOptimizer(self.learning_rate_ph, epsilon=1e-4)
        # Clip gradients
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online')
        grads_and_vars = opt.compute_gradients(self.total_error, online_vars)
        clipped_grads = [(tf.clip_by_norm(grad, clip_norm), var)
                         for grad, var in grads_and_vars if grad is not None]
        training_op = opt.apply_gradients(clipped_grads)

        return training_op

    def _build_target_update_op(self, alpha=1):
        # Get variables within defined scope
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'online')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
        # Create operations that copy the variables
        op_holder = [target_var.assign(alpha * online_var + (1 - alpha) * target_var)
                     for online_var, target_var in zip(online_vars, target_vars)]

        return op_holder

    def _create_summaries_op(self):
        self._maybe_create_writer(self.log_dir)
        tf.summary.scalar('loss', self.total_error)
        tf.summary.scalar('Q_mean', tf.reduce_mean(self.q_online_t))
        tf.summary.scalar('Q_max', tf.reduce_max(self.q_online_t))
        tf.summary.histogram('q_values', self.q_online_t)
        merged = tf.summary.merge_all()

        return merged

    def predict(self, sess, states):
        return sess.run(self.q_online_t, feed_dict={self.states_t_ph: states})

    def target_predict(self, sess, states):
        return sess.run(self.q_target_tp1, feed_dict={self.states_tp1_ph: states})

    def update_target_net(self, sess):
        sess.run(self.update_target_op)

    def write_summaries(self, sess, step, states_t, states_tp1, actions, rewards, dones):
        feed_dict = {
            self.states_t_ph: states_t,
            self.states_tp1_ph: states_tp1,
            self.actions_ph: actions,
            self.rewards_ph: rewards,
            self.dones_ph: dones
        }
        summary = sess.run(self.merged, feed_dict=feed_dict)
        self._writer.add_summary(summary, global_step=step)
