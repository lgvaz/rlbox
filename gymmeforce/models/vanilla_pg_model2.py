import numpy as np
import tensorflow as tf
from gymmeforce.models import BaseModel
# TODO: Maybe separete dir for graphs?
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.models.value_graphs import dense_value_graph
from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist
from gymmeforce.common.policy import Policy
from gymmeforce.common.data_gen import DataGenerator

class VanillaPGModel(BaseModel):
    def __init__(self, env_config, log_dir, use_baseline=True, entropy_coef=0., policy_graph=dense_policy_graph, value_graph=dense_value_graph):
        super().__init__(env_config, log_dir)
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef

        placeholders_config = {
            'states': [[None] + list(env_config['state_shape']), env_config['input_type']],
            'actions': [[None], tf.int32],
            'advantages': [[None], tf.float32],
            'vf_targets': [[None], tf.float32],
            'learning_rate': [[], tf.float32]
        }
        if env_config['action_space'] == 'discrete':
            placeholders_config['actions'] = [[None], tf.int32]
        elif env_config['action_space'] == 'continuous':
            placeholders_config['actions'] = [[None, env_config['num_actions']], tf.float32]

        self._create_placeholders(placeholders_config)
        self.policy = self._create_policy(self.placeholders['states'],
                                          self.placeholders['actions'],
                                          policy_graph)
        if self.use_baseline:
            self.baseline_sy = self._create_baseline(value_graph)
        self._add_losses()
        self._create_training_op(self.placeholders['learning_rate'])

    def _add_losses(self):
        ''' This method should be changed to add more losses'''
        self._pg_loss(self.policy, self.placeholders['advantages'], self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.placeholders['vf_targets'])

    def _pg_loss(self, policy, advantages, entropy_coef=0.1):
        loss = -tf.reduce_mean(policy.logprob_sy * advantages)
        loss += -(entropy_coef * policy.entropy_sy)
        tf.losses.add_loss(loss)

    def _baseline_loss(self, baseline_sy, targets):
        loss = tf.losses.mean_squared_error(labels=targets, predictions=baseline_sy)
        tf.losses.add_loss(loss)

    def _create_policy(self, states_ph, actions_ph, policy_graph):
        policy = Policy(self.env_config, states_ph, actions_ph, policy_graph)
        return policy

    def _create_baseline(self, value_graph):
        return value_graph(self.placeholders['states'])

    def select_action(self, sess, state):
        return self.policy.sample_action(sess, state[np.newaxis])

    def compute_baseline(self, sess, states):
        return sess.run(self.baseline_sy, feed_dict={self.placeholders['states']: states})

    def fit(self, sess, states, actions, vf_targets, advantages, num_epochs=10, batch_size=64, learning_rate=5e-3, logger=None):
        data = DataGenerator(states, actions, vf_targets, advantages)

        for i_epoch in range(num_epochs):
            data_iterator = data.iterate_once(batch_size)

            for b_states, b_actions, b_vf_targets, b_advantages in data_iterator:
                feed_dict = {
                    self.placeholders['states']: b_states,
                    self.placeholders['actions']: b_actions,
                    self.placeholders['vf_targets']: b_vf_targets,
                    self.placeholders['advantages']: b_advantages,
                    self.placeholders['learning_rate']: learning_rate
                }

                sess.run(self.training_op, feed_dict=feed_dict)

        if logger is not None:
            entropy = self.policy.entropy(sess, states)
            logger.add_log('Entropy', entropy, precision=3)
