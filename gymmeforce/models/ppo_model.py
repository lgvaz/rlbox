import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph
from gymmeforce.models.base_model import BaseModel
from gymmeforce.common.distributions import CategoricalDist

class PPOModel(BaseModel):
    def __init__(self, env_config, policy_graph=None, value_graph=None,
                 input_type=None, log_dir=None):
        super(PPOModel, self).__init__(env_config, log_dir)

        if policy_graph == None:
            policy_graph = dense_policy_graph
        if value_graph == None:
            value_graph = dense_value_graph

        self._create_placeholders()
        self.returns_mean, returns_var = tf.nn.moments(self.returns_ph, axes=[0])
        self.returns_std = returns_var ** 0.5
        self._build_value_graph(value_graph)
        self._build_policy_graph(policy_graph)

    def _create_placeholders(self):
        self.vf_target_ph = tf.placeholder(name='vf_target', shape=[None], dtype=tf.float32)
        # self.advantage_ph = tf.placeholder(name='advantage', shape=[None], dtype=tf.float32)
        self.returns_ph = tf.placeholder(name='returns', shape=[None], dtype=tf.float32)
        self.vf_lr_ph = tf.placeholder(name='vf_lr', shape=[], dtype=tf.float32)
        self.policy_lr_ph = tf.placeholder(name='policy_lr', shape=[], dtype=tf.float32)

    def _build_policy_graph(self, policy_graph):
        if self.env_config['action_space'] == 'discrete':
            # Create graph
            logits = policy_graph(self.states_t_ph, self.env_config)
            self.policy = CategoricalDist(logits)
            self.sample_action = self.policy.sample(tf.shape(self.states_t_ph)[0])
            logprob = self.policy.logprob(self.actions_ph)

        if self.env_config['action_space'] == 'continuous':
            # Create graph
            self.mean, self.logstd = policy_graph(self.states_t_ph, self.env_config)
            self.std = tf.exp(self.logstd)
            # Sample action
            self.sample_action = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
            self.sample_action = tf.clip_by_value(self.sample_action,
                                                  self.env_config['action_low_bound'],
                                                  self.env_config['action_high_bound'])
            # Calculate log probabilities
            logprob = -0.5 * tf.reduce_sum(self.logstd)
            logprob += -0.5 * tf.reduce_sum(((self.actions_ph - self.mean)
                                             / self.std) ** 2, axis=1)

        # Rescale baseline for same mean and variance of returns
        baseline = self.state_value
        baseline_mean, baseline_var = tf.nn.moments(baseline, axes=[0])
        baseline_std = baseline_var ** 0.5
        normalized_baseline = (baseline - baseline_mean) / (baseline_std + 1e-7)
        rescaled_baseline = normalized_baseline * self.returns_std + self.returns_mean
        advantages = self.returns_ph - rescaled_baseline
        # advantages = self.returns_ph - baseline
        self.policy_loss = -tf.reduce_sum(logprob * advantages)

        self.policy_update = tf.train.AdamOptimizer(self.policy_lr_ph).minimize(self.policy_loss)

    def _build_value_graph(self, value_graph):
        self.state_value = value_graph(self.states_t_ph, activation_fn=tf.nn.tanh)
        # Normalize target
        target = (self.returns_ph - self.returns_mean) / (self.returns_std + 1e-7)
        # target = self.state_value
        self.vf_loss = tf.losses.mean_squared_error(labels=target,
                                                    predictions=self.state_value)
        self.value_fn_update = tf.train.AdamOptimizer(self.vf_lr_ph).minimize(self.vf_loss)

    def select_action(self, sess, state):
        return sess.run(self.sample_action, feed_dict={self.states_t_ph: state[np.newaxis]})

    def predict_states_value(self, sess, states):
        return sess.run(self.state_value, feed_dict={self.states_t_ph: states})

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
