import numpy as np
import tensorflow as tf
from collections import namedtuple
from gymmeforce.models import BaseModel
from gymmeforce.models.policy_graphs import dense_policy_graph
from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist
# TODO: Maybe separete dir for graphs?

class VanillaPGModel(BaseModel):
    def __init__(self, env_config, log_dir, policy_graph=dense_policy_graph):
        super().__init__(env_config, log_dir)

        placeholders_config = {
            'states': [[None] + list(env_config['state_shape']), env_config['input_type']],
            'actions': [[None], tf.int32],
            'advantages': [[None], tf.float32],
            'vf_targets': [[None], tf.float32],
            'learning_rate': [[], tf.float32]
        }

        self._create_placeholders(placeholders_config)
        self.policy = self._create_policy(self.placeholders['states'],
                                          self.placeholders['actions'],
                                          policy_graph)
        self._add_losses()
        self._create_training_op(self.placeholders['learning_rate'])

    def _add_losses(self):
        ''' This method should be changed to add more losses'''
        self._pg_loss(self.policy, self.placeholders['advantages'])

    def _pg_loss(self, policy, advantages):
        pg_loss = -tf.reduce_mean(policy.logprob_sy * advantages)
        tf.losses.add_loss(pg_loss)

    def _create_policy(self, states_ph, actions_ph, policy_graph):
        logits = policy_graph(states_ph, self.env_config)

        policy = namedtuple('Policy', 'dist sample_action')
        policy.dist = CategoricalDist(logits)
        policy.logprob_sy = policy.dist.selected_logprob(actions_ph)
        policy.sample_action_sy = policy.dist.sample()
        policy.sample_action = lambda sess, states: sess.run(policy.sample_action_sy,
                                                             feed_dict={states_ph: states})
        policy.entropy_sy = policy.dist.entropy()
        policy.entropy = lambda sess, states: sess.run(policy.entropy_sy,
                                                       feed_dict={states_ph: states})

        return policy

    def select_action(self, sess, state):
        return self.policy.sample_action(sess, state[np.newaxis])

    def fit(self, sess, states, actions, advantages, learning_rate=5e-3):

        feed_dict = {
            self.placeholders['states']: states,
            self.placeholders['actions']: actions,
            self.placeholders['advantages']: advantages,
            self.placeholders['learning_rate']: learning_rate
        }

        sess.run(self.training_op, feed_dict=feed_dict)
