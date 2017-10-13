from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist


class Policy:
    def __init__(self, env_config, states_ph, actions_ph, graph,
                 scope='policy', reuse=None):
        self.states_ph = states_ph
        self.actions_ph = actions_ph

        params = graph(states_ph, env_config, scope=scope, reuse=reuse)
        if env_config['action_space'] == 'discrete':
            print('Making Discrete Policy')
            self.dist = CategoricalDist(params)
        elif env_config['action_space'] == 'continuous':
            print('Making Continuous Policy')
            self.dist = DiagGaussianDist(params,
                                         low_bound=env_config['action_low_bound'],
                                         high_bound=env_config['action_high_bound'])
        else:
            raise ValueError('{} action space not implemented'.format(
                env_config['action_space']))

        self.logprob_sy = self.dist.selected_logprob(actions_ph)
        self.sample_action_sy = self.dist.sample()
        self.entropy_sy = self.dist.entropy()

    def sample_action(self, sess, states):
        return sess.run(self.sample_action_sy, feed_dict={self.states_ph: states})

    def entropy(self, sess, states):
        return sess.run(self.entropy_sy, feed_dict={self.states_ph: states})
