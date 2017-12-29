import numpy as np
import tensorflow as tf

from rlbox.common.utils import tf_copy_params_op
from rlbox.models.base_model import BaseModel
from rlbox.models.q_graphs import deepmind_graph, simple_graph


class DQNModel(BaseModel):
    def __init__(self,
                 env_config,
                 graph=None,
                 double=False,
                 dueling=False,
                 gamma=0.99,
                 target_soft_update=1.,
                 grad_clip_norm=10,
                 **kwargs):
        super().__init__(env_config, grad_clip_norm=grad_clip_norm, **kwargs)
        self.double = double
        self.dueling = dueling
        self.gamma = gamma
        self.target_soft_update = target_soft_update

        # If input is an image defaults to deepmind_graph, else simple_graph
        if graph is None:
            if len(env_config['state_shape']) == 3:
                self.graph = deepmind_graph
                print('Using deepmind_graph')
            else:
                self.graph = simple_graph
                print('Using simple_graph')
        else:
            self.graph = graph
            print('Using custom graph')

        self._set_placeholders_config()
        self._create_placeholders(self.placeholders_config)

        self._create_graphs()

        self._build_optimization()
        self._build_target_update_op()

        # Create collections for loading later
        tf.add_to_collection('state_input', self.placeholders['states_t'])
        tf.add_to_collection('q_online_t', self.q_online_t)

    def _set_placeholders_config(self):
        self.placeholders_config = {
            'states_t': [[None] + list(self.env_config['state_shape']),
                         self.env_config['input_type']],
            'states_tp1': [[None] + list(self.env_config['state_shape']),
                           self.env_config['input_type']],
            'actions': [[None], tf.int32],
            'rewards': [[None], tf.float32],
            'dones': [[None], tf.float32],
            'n_step': [[], tf.float32],
            'learning_rate': [[], tf.float32]
        }

    def _create_graphs(self):
        if tf.uint8 == self.env_config['input_type']:
            # Convert to float on GPU
            states_t = tf.cast(self.placeholders['states_t'], tf.float32) / 255.
            states_tp1 = tf.cast(self.placeholders['states_tp1'], tf.float32) / 255.
        else:
            states_t = self.placeholders['states_t']
            states_tp1 = self.placeholders['states_tp1']

        self.q_online_t = self.graph(
            states_t, self.env_config['num_actions'], 'online', dueling=self.dueling)
        self.q_target_tp1 = self.graph(
            states_tp1, self.env_config['num_actions'], 'target', dueling=self.dueling)
        if self.double:
            self.q_online_tp1 = self.graph(
                states_tp1,
                self.env_config['num_actions'],
                'online',
                dueling=self.dueling,
                reuse=True)

    def _build_optimization(self):
        # Choose only the q values for selected actions
        onehot_actions = tf.one_hot(self.placeholders['actions'],
                                    self.env_config['num_actions'])
        q_t = tf.reduce_sum(self.q_online_t * onehot_actions, axis=1)

        # Caculate td_target
        if self.double:
            best_actions_onehot = tf.one_hot(
                tf.argmax(self.q_online_tp1, axis=1), self.env_config['num_actions'])
            q_tp1 = tf.reduce_sum(self.q_target_tp1 * best_actions_onehot, axis=1)
        else:
            q_tp1 = tf.reduce_max(self.q_target_tp1, axis=1)

        td_target = (self.placeholders['rewards'] + (1 - self.placeholders['dones']) *
                     (self.gamma**self.placeholders['n_step']) * q_tp1)
        tf.losses.huber_loss(labels=td_target, predictions=q_t)

        # Create training operation
        opt_config = dict(epsilon=1e-4)
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online')
        self._create_training_op(
            learning_rate=self.placeholders['learning_rate'],
            var_list=online_vars,
            opt_config=opt_config)

    def _build_target_update_op(self):
        self.update_target_op = tf_copy_params_op('online', 'target',
                                                  self.target_soft_update)

    def _create_summaries_op(self):
        super()._create_summaries_op()
        tf.summary.scalar('network/Q_mean', tf.reduce_mean(self.q_online_t))
        tf.summary.scalar('network/Q_max', tf.reduce_max(self.q_online_t))
        tf.summary.histogram('network/q_values', self.q_online_t)

    def predict(self, sess, states):
        return sess.run(
            self.q_online_t, feed_dict={
                self.placeholders['states_t']: states
            })

    def target_predict(self, sess, states):
        return sess.run(
            self.q_target_tp1, feed_dict={
                self.placeholders['states_tp1']: states
            })

    def update_target_net(self, sess):
        sess.run(self.update_target_op)

    def fit(self, sess, batch, learning_rate):
        batch['learning_rate'] = learning_rate
        self._fetch_placeholders_data_dict(batch)
        sess.run(self.training_op, feed_dict=self.placeholders_and_data)

    def write_logs(self, sess, logger=None):
        self._write_summaries(sess, self.placeholders_and_data)
