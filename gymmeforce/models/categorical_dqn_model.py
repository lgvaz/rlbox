import numpy as np
import tensorflow as tf

from gymmeforce.models import DQNModel
from gymmeforce.common.tf_utils import slice_2nd_dim


class CategoricalDQNModel(DQNModel):
    def __init__(self, env_config, num_atoms=51, vmax=10., vmin=-10., **kwargs):
        self.num_atoms = num_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.atoms = tf.lin_space(self.vmin, self.vmax, self.num_atoms)
        super().__init__(env_config, **kwargs)
        self.expected_q_values = tf.tensordot(self.q_online_t, self.atoms, axes=1)

    def _get_graph_config(self):
        graph_config = dict(
            output_size=(self.env_config['num_actions'], self.num_atoms),
            dueling=self.dueling)

        return graph_config

    def predict(self, sess, states):
        feed_dict = {self.placeholders['states_t']: states}
        return sess.run(self.expected_q_values, feed_dict=feed_dict)

    def _build_optimization(self):
        # TODO: hardcoded batch_size
        batch_size = 32
        # Choose only the q values for selected actions
        q_dist_t = slice_2nd_dim(
            self.q_online_t, self.placeholders['actions'], batch_size=batch_size)

        # Weighted sum of atoms based on their probability (output of the network)
        expected_q_tp1 = tf.tensordot(self.q_target_tp1, self.atoms, axes=1)
        # Select max action based on expected value (same as Q-learning)
        best_actions_tp1 = tf.argmax(expected_q_tp1, axis=-1)

        # Select the atom probabilities of the best action
        q_dist_tp1 = slice_2nd_dim(
            self.q_target_tp1, best_actions_tp1, batch_size=batch_size)

        # Calculate target distribution
        rewards = tf.expand_dims(self.placeholders['rewards'], axis=1)
        dones = tf.expand_dims(self.placeholders['dones'], axis=-1)
        td_target = rewards + (1 - dones) * self.gamma * q_dist_tp1
        # Clip values so that they don't fall outside distribution
        # Use a small epsilon to avoid ceil and floor to be the same number
        # because ceil and floor of an int is a int
        td_target = tf.clip_by_value(td_target, self.vmin + 1e-5, self.vmax - 1e-5)

        # Indices of lower and upper atoms
        b = (td_target - self.vmin) / (self.atoms[1] - self.atoms[0])
        l = tf.floor(b)
        u = tf.ceil(b)
        # Convert to one dimension
        batches_ids = tf.expand_dims(
            tf.cast(tf.range(batch_size) * self.num_atoms, tf.float32), axis=-1)
        b_ravel = tf.reshape(batches_ids + b, [-1])
        l_ravel = tf.reshape(batches_ids + l, [-1])
        u_ravel = tf.reshape(batches_ids + u, [-1])
        q_dist_t_ravel = tf.reshape(q_dist_t, [-1])
        q_dist_tp1_ravel = tf.reshape(q_dist_tp1, [-1])

        # Split weight between neighboring atoms
        lower_atoms = tf.gather(q_dist_tp1_ravel, tf.cast(l_ravel, tf.int32))
        upper_atoms = tf.gather(q_dist_tp1_ravel, tf.cast(u_ravel, tf.int32))
        split_lower = lower_atoms * (u_ravel - b_ravel)
        split_upper = upper_atoms * (b_ravel - l_ravel)

        # WHAT??
        q_dist_t_softmax = tf.clip_by_value(tf.nn.softmax(q_dist_t), 0.01, 0.99)
        q_dist_t_softmax_ravel = tf.reshape(q_dist_t_softmax, [-1])

        # loss_lower = -tf.reduce_sum(split_lower * tf.log(q_dist_t_softmax_ravel))
        # loss_upper = -tf.reduce_sum(split_upper * tf.log(q_dist_t_softmax_ravel))

        loss_lower = -tf.reduce_sum(split_lower * q_dist_t_ravel)
        loss_upper = -tf.reduce_sum(split_upper * q_dist_t_ravel)

        tf.losses.add_loss(loss_lower + loss_upper)
        # tf.losses.add_loss(loss_upper)

        opt_config = dict(epsilon=1e-4)
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online')
        self._create_training_op(
            learning_rate=self.placeholders['learning_rate'],
            var_list=online_vars,
            opt_config=opt_config)
