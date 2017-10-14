import numpy as np
import tensorflow as tf
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph
from gymmeforce.models.base_model import BaseModel
from gymmeforce.common.policy import Policy
from gymmeforce.common.data_gen import DataGenerator


class VanillaPGModel(BaseModel):
    def __init__(self, env_config, normalize_advantages=False, use_baseline=True,
                 normalize_baseline=False, entropy_coef=0., policy_graph=None,
                 value_graph=None, **kwargs):
        super(VanillaPGModel, self).__init__(env_config, **kwargs)
        self.normalize_advantages = normalize_advantages
        self.use_baseline = use_baseline
        self.normalize_baseline = normalize_baseline
        self.entropy_coef = entropy_coef

        if policy_graph == None:
            policy_graph = dense_policy_graph
        if value_graph == None:
            value_graph = dense_value_graph

        self._set_placeholders_config()
        self._create_placeholders(self.placeholders_config)
        self.policy = self._create_policy(self.placeholders['states'],
                                          self.placeholders['actions'],
                                          policy_graph)

        if self.use_baseline:
            self.baseline_sy = self._create_baseline(value_graph)
            self.baseline_target = self.placeholders['returns']
            self.baseline = self.placeholders['baseline']
            if normalize_baseline:
                self._normalize_baseline()

        self._add_losses()
        self._create_training_op(self.placeholders['learning_rate'])

    def _set_placeholders_config(self):
        ''' Modify this method to add new placeholders '''
        self.placeholders_config = {
            'states': [[None] + list(self.env_config['state_shape']),
                       self.env_config['input_type']],
            'returns': [[None], tf.float32],
            'baseline': [[None], tf.float32],
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
        advantages = self._estimate_advatanges()
        loss = -tf.reduce_mean(policy.logprob_sy * advantages)
        tf.losses.add_loss(loss)

    def _entropy_loss(self, policy, entropy_coef):
        loss = -(entropy_coef * policy.entropy_sy)
        tf.losses.add_loss(loss)

    def _estimate_advatanges(self):
        if self.use_baseline:
            advantages = self.placeholders['returns'] - self.baseline
        else:
            advantages = self.placeholders['returns']

        if self.normalize_advantages:
            advs_mean, advs_var = tf.nn.moments(advantages, axes=[0])
            advs_std = advs_var ** 0.5
            advantages = (advantages - advs_mean) / (advs_std + 1e-7)

        return advantages

    def _baseline_loss(self, baseline_sy, targets):
        loss = tf.losses.mean_squared_error(labels=targets, predictions=baseline_sy)
        tf.losses.add_loss(loss)

    def _create_baseline(self, value_graph):
        return value_graph(self.placeholders['states'])

    def _create_policy(self, states_ph, actions_ph, policy_graph,
                       scope='policy', reuse=None):
        policy = Policy(self.env_config, states_ph, actions_ph, policy_graph)
        return policy

    def _normalize_baseline(self):
        # Normalize target values for baseline
        returns_mean, returns_var = tf.nn.moments(self.placeholders['returns'], axes=[0])
        returns_std = returns_var ** 0.5
        self.baseline_target = (self.placeholders['returns'] - returns_mean) / (returns_std + 1e-7)

        # Rescale baseline for same mean and variance of returns
        baseline_mean, baseline_var = tf.nn.moments(self.baseline, axes=[0])
        baseline_std = baseline_var ** 0.5
        normalized_baseline = (self.baseline - baseline_mean) / (baseline_std + 1e-7)
        self.baseline = normalized_baseline * returns_std + returns_mean

    def _fetch_placeholders_data_dict(self, sess, states, actions, returns):
        '''
        Create a dictionary mapping placeholders to their correspondent value
        Modify this method to include new placeholders to feed_dict used by training_op
        '''
        self.placeholders_and_data = {
            self.placeholders['states']: states,
            self.placeholders['actions']: actions,
            self.placeholders['returns']: returns
        }
        # Calculate baseline
        if self.use_baseline:
            baseline = self.compute_baseline(sess, states)
            self.placeholders_and_data[self.placeholders['baseline']] = baseline

    def _create_summaries_op(self):
        self._maybe_create_writer()

        if self.env_config['action_space'] == 'discrete':
            tf.summary.histogram('action_probs', self.policy.dist.action_probs)

        if self.env_config['action_space'] == 'continuous':
            means = self.policy.dist.mean
            stds = self.policy.dist.std
            tf.summary.histogram('dist_means', means)
            tf.summary.histogram('dist_standard_devs', stds)
            tf.summary.scalar('dist_means_mean', tf.reduce_mean(means))
            tf.summary.scalar('dist_standard_devs_mean', tf.reduce_mean(stds))

        merged = tf.summary.merge_all()

        return merged

    def write_summaries(self, sess, feed_dict):
        if self.merged is None:
            self.merged = self._create_summaries_op()
        summary = sess.run(self.merged, feed_dict=feed_dict)
        self._writer.add_summary(summary, self.get_global_step(sess))

    def select_action(self, sess, state):
        return self.policy.sample_action(sess, state[np.newaxis])

    def compute_baseline(self, sess, states):
        return sess.run(self.baseline_sy, feed_dict={self.placeholders['states']: states})

    def fit(self, sess, states, actions, returns, learning_rate,
            num_epochs=10, batch_size=64, logger=None):
        self._fetch_placeholders_data_dict(sess, states, actions, returns)
        data = DataGenerator(self.placeholders_and_data)

        for i_epoch in range(num_epochs):
            for feed_dict in data.fetch_batch_dict(batch_size):
                feed_dict[self.placeholders['learning_rate']] = learning_rate
                loss, _ = sess.run([self.loss_sy, self.training_op], feed_dict=feed_dict)

        self.write_summaries(sess, self.placeholders_and_data)
