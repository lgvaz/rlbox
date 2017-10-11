import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from collections import namedtuple
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph
from gymmeforce.models.base_model import BaseModel
from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist

class VanillaPGModel(BaseModel):
    def __init__(self, env_config, normalize_baseline=True,
                 policy_graph=None, value_graph=None, input_type=None, log_dir=None):
        super(VanillaPGModel, self).__init__(env_config, log_dir)

        if policy_graph == None:
            policy_graph = dense_policy_graph
        if value_graph == None:
            value_graph = dense_value_graph

        placeholders_config = {
            'states': [[None] + list(env_config['state_shape']), env_config['input_type']],
            'returns': [[None], tf.float32],
            'baseline': [[None], tf.float32],
            'advantages': [[None], tf.float32],
            'value_fn_lr': [[], tf.float32],
            'policy_lr': [[], tf.float32]
        }
        if env_config['action_space'] == 'discrete':
            placeholders_config['actions'] = [[None], tf.int32]
        if env_config['action_space'] == 'continuous':
            placeholders_config['actions'] = [[None, env_config['num_actions']], tf.float32]

        self._create_placeholders(placeholders_config)
        self.value_fn = self._create_value_fn(self.placeholders['states'],
                                              value_graph)
        self.policy = self._create_policy(self.placeholders['states'],
                                          self.placeholders['actions'],
                                          policy_graph)
        self.vf_target = self.placeholders['returns']
        self.baseline = self.placeholders['baseline']
        if normalize_baseline:
            self._normalize_baseline()
        self.value_fn_update = self._create_value_fn_training_op(self.placeholders['value_fn_lr'])
        self.policy_update = self._create_policy_training_op(self.policy,
                                                             self.placeholders['policy_lr'])

    def _create_policy_training_op(self, policy, learning_rate):
        '''
        This method should be changed to add new losses (e.g. KL penalty)
        '''
        pg_loss = self._pg_loss(policy)
        self.pg_loss = pg_loss
        training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(pg_loss)
        return training_op

    def _pg_loss(self, policy):
        advantages = self.placeholders['returns'] - self.baseline
        # # Normalize advantages
        # advs_mean, advs_var = tf.nn.moments(advantages, axes=[0])
        # advs_std = advs_var ** 0.5
        # norm_advantages = (advantages - advs_mean) / (advs_std + 1e-7)

        pg_loss = -tf.reduce_mean(policy.logprob * self.placeholders['advantages'])

        return pg_loss

    def _create_value_fn(self, states_ph, value_graph):
        state_value = value_graph(states_ph, activation_fn=tf.nn.tanh)

        baseline = namedtuple('Value_fn', 'state_value')
        baseline.state_value = state_value

        return baseline

    def _create_policy(self, states_ph, actions_ph, policy_graph, scope='policy', reuse=None):
        if self.env_config['action_space'] == 'discrete':
            logits = policy_graph(states_ph, self.env_config, scope=scope, reuse=reuse)

            policy = namedtuple('Policy', 'dist sample_action logprob entropy')
            policy.dist = CategoricalDist(logits)
            policy.sample_action = policy.dist.sample(tf.shape(states_ph)[0])
            policy.logprob = policy.dist.selected_logprob(actions_ph)
            policy.entropy = policy.dist.entropy()

            return policy

        if self.env_config['action_space'] == 'continuous':
            # Create graph
            mean, logstd = policy_graph(states_ph, self.env_config, scope=scope, reuse=reuse)

            policy = namedtuple('Policy', 'dist sample_action logprob entropy')
            policy.dist = DiagGaussianDist(mean, logstd,
                                           low_bound=self.env_config['action_low_bound'],
                                           high_bound=self.env_config['action_high_bound'])
            policy.sample_action = policy.dist.sample()
            policy.logprob = policy.dist.logprob(actions_ph)
            policy.entropy = policy.dist.entropy()

            return policy

    def _normalize_baseline(self):
        # Normalize target values for baseline
        returns_mean, returns_var = tf.nn.moments(self.placeholders['returns'], axes=[0])
        returns_std = returns_var ** 0.5
        self.vf_target = (self.placeholders['returns'] - returns_mean) / (returns_std + 1e-7)

        # Rescale baseline for same mean and variance of returns
        baseline_mean, baseline_var = tf.nn.moments(self.baseline, axes=[0])
        baseline_std = baseline_var ** 0.5
        normalized_baseline = (self.baseline - baseline_mean) / (baseline_std + 1e-7)
        self.baseline = normalized_baseline * returns_std + returns_mean

    def _create_value_fn_training_op(self, learning_rate):
        vf_loss = tf.losses.mean_squared_error(labels=self.vf_target,
                                               predictions=self.value_fn.state_value)
        training_op = tf.train.AdamOptimizer(learning_rate).minimize(vf_loss)
        return training_op

    def select_action(self, sess, state):
        return sess.run(self.policy.sample_action,
                        feed_dict={self.placeholders['states']: state[np.newaxis]})

    def predict_states_value(self, sess, states):
        return sess.run(self.value_fn.state_value, feed_dict={self.placeholders['states']: states})

    # TODO: Compute adv and then fit policy
    def train(self, sess, states, actions, returns, policy_learning_rate, vf_learning_rate,
                    num_epochs=10, batch_size=64, logger=None):
        # Calculate baseline
        feed_dict = {self.placeholders['states']: states}
        baseline = sess.run(self.value_fn.state_value, feed_dict=feed_dict)
        advs = returns - baseline
        # Split data into multiple batches and train on multiple epochs
        num_batches = max(np.shape(states)[0] // batch_size, 1)
        batch_size = np.shape(states)[0] // num_batches

        mean_old, logstd_old, std_old, entropy_old = sess.run([self.policy.dist.mean, self.policy.dist.logstd, self.policy.dist.std, self.policy.entropy], feed_dict=feed_dict)
        for i_epoch in range(num_epochs):
            states, actions, returns, baseline, advs = shuffle(states, actions, returns, baseline, advs)
            for i_batch in range(num_batches):
                start = i_batch * batch_size
                end = (i_batch + 1) * batch_size
                states_batch = states[start : end]
                actions_batch = actions[start : end]
                returns_batch = returns[start : end]
                baseline_batch = baseline[start : end]
                advs_batch = advs[start : end]

                feed_dict = {
                    self.placeholders['states']: states_batch,
                    self.placeholders['actions']: actions_batch,
                    self.placeholders['returns']: returns_batch,
                    self.placeholders['baseline']: baseline_batch,
                    self.placeholders['advantages']: advs_batch,
                    self.placeholders['value_fn_lr']: vf_learning_rate,
                    self.placeholders['policy_lr']: policy_learning_rate
                }

                pg_loss = sess.run([self.pg_loss,
                                    self.policy_update,
                                    self.value_fn_update],
                                   feed_dict=feed_dict)[0]
                logger.add_log('pg_loss', pg_loss)

        pg_loss, mean, logstd, std, entropy = sess.run([self.pg_loss, self.policy.dist.mean, self.policy.dist.logstd, self.policy.dist.std, self.policy.entropy], feed_dict=feed_dict)

        if logger:
            logger.add_log('returns', np.mean(returns))
            logger.add_log('baseline', np.mean(baseline))
            logger.add_log('adv', np.mean(advs))
            logger.add_log('Entropy_old', entropy_old)
            logger.add_log('Entropy', entropy)
            logger.add_log('mean_old', np.mean(mean_old))
            logger.add_log('mean', np.mean(mean))
            logger.add_log('Logstd_old', logstd_old)
            logger.add_log('Logstd', logstd)
            logger.add_log('std_old', std_old)
            logger.add_log('std', std)
        if logger:
            logger.add_log('Policy Learning Rate', policy_learning_rate, precision=5)
            logger.add_log('Baseline Learning Rate', vf_learning_rate, precision=5)