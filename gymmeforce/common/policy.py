import tensorflow as tf
from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist


class Policy:
    def __init__(self, env_config, states_ph, actions_ph, graph,
                 scope='policy', reuse=None, trainable=True):
        with tf.variable_scope(scope):
            self.states_ph = states_ph
            self.actions_ph = actions_ph

            params = graph(states_ph, env_config, scope='graph', reuse=reuse, trainable=trainable)
            if env_config['action_space'] == 'discrete':
                print('Making Discrete Policy with scope ({})'.format(scope))
                self.dist_function = CategoricalDist
                self.dist = CategoricalDist(params)
            elif env_config['action_space'] == 'continuous':
                print('Making Continuous Policy with scope ({})'.format(scope))
                self.dist_function = DiagGaussianDist
                self.dist = DiagGaussianDist(params,
                                            low_bound=env_config['action_low_bound'],
                                            high_bound=env_config['action_high_bound'])
            else:
                raise ValueError('{} action space not implemented'.format(
                    env_config['action_space']))

            with tf.variable_scope('logprob'):
                self.logprob_sy = self.dist.selected_logprob(actions_ph)
            with tf.variable_scope('sample_action'):
                self.sample_action_sy = self.dist.sample()
            with tf.variable_scope('entropy'):
                self.entropy_sy = self.dist.entropy()

    def sample_action(self, sess, states):
        return sess.run(self.sample_action_sy, feed_dict={self.states_ph: states})

    def entropy(self, sess, states):
        return sess.run(self.entropy_sy, feed_dict={self.states_ph: states})

    def kl_divergence(self, old_policy, new_policy):
        with tf.variable_scope('kl_divergence'):
            return self.dist_function.kl_divergence(old_policy.dist, new_policy.dist)
