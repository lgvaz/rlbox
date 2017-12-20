import numpy as np
import tensorflow as tf

from gymmeforce.common.data_gen import DataGenerator
from gymmeforce.common.policy import Policy
from gymmeforce.models.base_model import BaseModel
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph


class VanillaPGModel(BaseModel):
    def __init__(self,
                 env_config,
                 use_baseline=True,
                 entropy_coef=0.0,
                 policy_graph=None,
                 value_graph=None,
                 **kwargs):
        super(VanillaPGModel, self).__init__(env_config, **kwargs)
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        self.policy_graph = policy_graph or dense_policy_graph
        self.value_graph = value_graph or dense_value_graph

        self._set_placeholders_config()
        self._create_placeholders(self.placeholders_config)
        self._create_policy()

        if self.use_baseline:
            self.baseline_sy = self._create_baseline()
            self.baseline_target = self.placeholders['returns']

        self._add_losses()
        self._create_training_op(
            self.placeholders['learning_rate'], opt_config=dict(epsilon=1e-5))

    def _set_placeholders_config(self):
        ''' Modify this method to add new placeholders '''
        self.placeholders_config = {
            'states': [[None] + list(self.env_config['state_shape']),
                       self.env_config['input_type']],
            'returns': [[None], tf.float32],
            'advantages': [[None], tf.float32],
            'learning_rate': [[], tf.float32]
        }
        if self.env_config['action_space'] == 'discrete':
            self.placeholders_config['actions'] = [[None], tf.int32]
        if self.env_config['action_space'] == 'continuous':
            self.placeholders_config['actions'] = [[None, self.env_config['num_actions']],
                                                   tf.float32]

    def _add_losses(self):
        ''' Modify this method to add new losses e.g. KL penalty '''
        self._pg_loss()
        self._entropy_loss()
        if self.use_baseline:
            self._baseline_loss()

    def _pg_loss(self):
        with tf.variable_scope('pg_loss'):
            loss = -tf.reduce_mean(
                self.policy.logprob_sy * self.placeholders['advantages'])
            tf.losses.add_loss(loss)

    def _entropy_loss(self):
        with tf.variable_scope('entropy_loss'):
            loss = -(self.entropy_coef * self.policy.entropy_sy)
            tf.losses.add_loss(loss)

    def _baseline_loss(self):
        with tf.variable_scope('baseline_loss'):
            # This automatically adds the loss to the loss collection
            loss = tf.losses.mean_squared_error(
                labels=self.baseline_target, predictions=self.baseline_sy)

    def _create_baseline(self):
        return self.value_graph(self.placeholders['states'])

    def _create_policy(self):
        self.policy = Policy(
            self.env_config,
            states_ph=self.placeholders['states'],
            actions_ph=self.placeholders['actions'],
            graph=self.policy_graph,
            scope='policy')

    def _create_summaries_op(self):
        super()._create_summaries_op()
        tf.summary.histogram('policy/logprob', self.policy.logprob_sy)
        tf.summary.scalar('policy/logprob/mean', tf.reduce_mean(self.policy.logprob_sy))

        advantages = self.placeholders['advantages']
        tf.summary.histogram('advantages', advantages)
        tf.summary.scalar('advantages/max', tf.reduce_max(advantages))
        tf.summary.scalar('advantages/min', tf.reduce_min(advantages))

        if self.use_baseline:
            tf.summary.histogram('baseline', self.baseline_sy)
            tf.summary.scalar('baseline/mean', tf.reduce_mean(self.baseline_sy))

        if self.env_config['action_space'] == 'discrete':
            tf.summary.histogram('policy/action_probs', self.policy.dist.action_probs_sy)

        if self.env_config['action_space'] == 'continuous':
            means = self.policy.dist.mean
            stds = self.policy.dist.std
            tf.summary.histogram('policy/means', means)
            tf.summary.histogram('policy/standard_devs', stds)
            tf.summary.scalar('policy/means/mean', tf.reduce_mean(means))
            tf.summary.scalar('policy/standard_devs/mean', tf.reduce_mean(stds))

    def write_logs(self, sess, logger):
        entropy = sess.run(self.policy.entropy_sy, feed_dict=self.placeholders_and_data)
        logger.add_log('policy/Entropy', entropy)

        self._write_summaries(sess, self.placeholders_and_data)

    def select_action(self, sess, state):
        return self.policy.sample_action(sess, state[np.newaxis])

    def compute_baseline(self, sess, states):
        return sess.run(self.baseline_sy, feed_dict={self.placeholders['states']: states})

    def fit(self, sess, batch, learning_rate, num_epochs=10, batch_size=64, **kwargs):
        self._fetch_placeholders_data_dict(batch)
        data = DataGenerator(self.placeholders_and_data)

        for i_epoch in range(num_epochs):
            for feed_dict in data.fetch_batch_dict(batch_size):
                feed_dict[self.placeholders['learning_rate']] = learning_rate
                sess.run([self.training_op], feed_dict=feed_dict)

            for callback in self.callbacks:
                if callback(sess):
                    return
