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

        self._create_placeholders()
        self.value_fn = self._create_value_fn(value_graph)
        self.policy = self._create_policy(self.states_t_ph, self.actions_ph, policy_graph)
        self.vf_target = self.returns_ph
        self.baseline = self.value_fn.state_value
        if normalize_baseline:
            self._normalize_baseline()
        self.value_fn_update = self._create_value_fn_training_op(self.vf_lr_ph)
        self.policy_update = self._create_policy_training_op(self.policy, self.policy_lr_ph)

    def _create_policy_training_op(self, policy, learning_rate):
        '''
        This method should be changed to add new losses (e.g. KL penalty)
        '''
        pg_loss = self._pg_loss(policy)
        training_op = tf.train.AdamOptimizer(learning_rate).minimize(pg_loss)
        return training_op

    def _pg_loss(self, policy):
        advantages = self.returns_ph - self.baseline
        pg_loss = -tf.reduce_sum(policy.logprob * advantages)
        return pg_loss

    def _create_placeholders(self):
        self.vf_target_ph = tf.placeholder(name='vf_target', shape=[None], dtype=tf.float32)
        # self.advantage_ph = tf.placeholder(name='advantage', shape=[None], dtype=tf.float32)
        self.returns_ph = tf.placeholder(name='returns', shape=[None], dtype=tf.float32)
        self.vf_lr_ph = tf.placeholder(name='vf_lr', shape=[], dtype=tf.float32)
        self.policy_lr_ph = tf.placeholder(name='policy_lr', shape=[], dtype=tf.float32)

    def _create_value_fn(self, value_graph):
        state_value = value_graph(self.states_t_ph, activation_fn=tf.nn.tanh)

        baseline = namedtuple('Value_fn', 'state_value')
        baseline.state_value = state_value

        return baseline

    def _create_policy(self, states_ph, actions_ph, policy_graph, scope='policy', reuse=None):
        if self.env_config['action_space'] == 'discrete':
            logits = policy_graph(states_ph, self.env_config, scope=scope, reuse=None)

            policy = namedtuple('Policy', 'dist sample_action logprob')
            policy.dist = CategoricalDist(logits)
            policy.sample_action = policy.dist.sample(tf.shape(states_ph)[0])
            policy.logprob = policy.dist.logprob(actions_ph)

            return policy

        if self.env_config['action_space'] == 'continuous':
            # Create graph
            mean, logstd = policy_graph(self.states_t_ph, self.env_config)

            policy = namedtuple('Policy', 'dist sample_action logprob')
            policy.dist = DiagGaussianDist(mean, logstd,
                                           low_bound=self.env_config['action_low_bound'],
                                           high_bound=self.env_config['action_high_bound'])
            policy.sample_action = policy.dist.sample()
            policy.logprob = policy.dist.logprob(self.actions_ph)

            return policy

    def _normalize_baseline(self):
        # Normalize target values for baseline
        returns_mean, returns_var = tf.nn.moments(self.returns_ph, axes=[0])
        returns_std = returns_var ** 0.5
        self.vf_target = (self.returns_ph - returns_mean) / (returns_std + 1e-7)

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
                        feed_dict={self.states_t_ph: state[np.newaxis]})

    def predict_states_value(self, sess, states):
        return sess.run(self.value_fn.state_value, feed_dict={self.states_t_ph: states})

    def train(self, sess, states, actions, returns, policy_learning_rate, vf_learning_rate,
                    num_epochs=10, batch_size=64):
        # Split data into multiple batches and train on multiple epochs
        num_batches = max(np.shape(states)[0] // batch_size, 1)
        batch_size = np.shape(states)[0] // num_batches

        for i_epoch in range(num_epochs):
            states, actions, returns = shuffle(states, actions, returns)
            for i_batch in range(num_batches):
                start = i_batch * batch_size
                end = (i_batch + 1) * batch_size
                states_batch = states[start : end]
                actions_batch = actions[start : end]
                # vf_targets_batch = vf_targets[start : end]
                returns_batch = returns[start : end]

                feed_dict = {
                    self.states_t_ph: states_batch,
                    self.actions_ph: actions_batch,
                    # self.vf_target_ph: vf_targets_batch,
                    self.returns_ph: returns_batch,
                    self.vf_lr_ph: vf_learning_rate,
                    self.policy_lr_ph: policy_learning_rate,
                }

                sess.run([self.policy_update, self.value_fn_update], feed_dict=feed_dict)
