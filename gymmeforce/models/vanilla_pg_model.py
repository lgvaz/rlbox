import numpy as np
import tensorflow as tf
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph
from gymmeforce.models.base_model import BaseModel
from gymmeforce.common.policy import Policy
from gymmeforce.common.data_gen import DataGenerator


class VanillaPGModel(BaseModel):
    def __init__(self, env_config, use_baseline=True, entropy_coef=0.0,
                 policy_graph=None, value_graph=None, **kwargs):
        super(VanillaPGModel, self).__init__(env_config, **kwargs)
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef

        if policy_graph == None:
            self.policy_graph = dense_policy_graph
        if value_graph == None:
            self.value_graph = dense_value_graph

        self._set_placeholders_config()
        self._create_placeholders(self.placeholders_config)
        self._create_policy(self.placeholders['states'],
                            self.placeholders['actions'],
                            self.policy_graph)

        if self.use_baseline:
            self.baseline_sy = self._create_baseline(self.value_graph)
            self.baseline_target = self.placeholders['returns']

        self._add_losses()
        self._create_training_op(self.placeholders['learning_rate'])

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
        self._pg_loss(self.policy)
        self._entropy_loss(self.policy, self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.baseline_target)

    def _pg_loss(self, policy):
        with tf.variable_scope('pg_loss'):
            loss = -tf.reduce_mean(policy.logprob_sy * self.placeholders['advantages'])
            tf.losses.add_loss(loss)

    def _entropy_loss(self, policy, entropy_coef):
        with tf.variable_scope('entropy_loss'):
            loss = -(entropy_coef * policy.entropy_sy)
            tf.losses.add_loss(loss)

    def _baseline_loss(self, baseline_sy, targets):
        with tf.variable_scope('baseline_loss'):
            # This automatically adds the loss to the loss collection
            loss = tf.losses.mean_squared_error(labels=targets, predictions=baseline_sy)

    def _create_baseline(self, value_graph):
        return value_graph(self.placeholders['states'])

    def _create_policy(self, states_ph, actions_ph, policy_graph,
                       scope='policy', reuse=None):
        self.policy = Policy(self.env_config, states_ph, actions_ph, policy_graph)

    def _fetch_placeholders_data_dict(self, sess, states, actions, returns, advantages):
        '''
        Create a dictionary mapping placeholders to their correspondent value
        Modify this method to include new placeholders to feed_dict used by training_op
        '''
        self.placeholders_and_data = {
            self.placeholders['states']: states,
            self.placeholders['actions']: actions,
            self.placeholders['returns']: returns,
            self.placeholders['advantages']: advantages
        }

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

        if self.use_baseline:
            y_pred = sess.run(self.baseline_sy, feed_dict=self.placeholders_and_data)
            y_true = self.placeholders_and_data[self.placeholders['returns']]
            explained_variance = np.var(y_true - y_pred) / np.var(y_true)
            logger.add_log('baseline/Explained Variance', explained_variance)

        self.write_summaries(sess, self.placeholders_and_data)

    def write_summaries(self, sess, feed_dict):
        if self.merged is None:
            self._create_summaries_op()
            self.merged = tf.summary.merge_all()
        summary = sess.run(self.merged, feed_dict=feed_dict)
        self._writer.add_summary(summary, self.get_global_step(sess))

    def select_action(self, sess, state):
        return self.policy.sample_action(sess, state[np.newaxis])

    def compute_baseline(self, sess, states):
        return sess.run(self.baseline_sy, feed_dict={self.placeholders['states']: states})

    def fit(self, sess, states, actions, returns, advantages, learning_rate,
            num_epochs=10, batch_size=64, logger=None):
        self._fetch_placeholders_data_dict(sess, states, actions, returns, advantages)
        data = DataGenerator(self.placeholders_and_data)

        for i_epoch in range(num_epochs):
            for feed_dict in data.fetch_batch_dict(batch_size):
                feed_dict[self.placeholders['learning_rate']] = learning_rate
                sess.run([self.training_op], feed_dict=feed_dict)
