from gymmeforce.common.distributions import CategoricalDist, DiagGaussianDist


class Policy:
    def __init__(self, env_config, states_ph, actions_ph, graph):
        self.states_ph = states_ph
        self.actions_ph = actions_ph

        self.logits = graph(states_ph, env_config)
        self.dist = CategoricalDist(self.logits)

        self.logprob_sy = self.dist.selected_logprob(actions_ph)
        self.sample_action_sy = self.dist.sample()
        self.entropy_sy = self.dist.entropy()

    def sample_action(self, sess, states):
        return sess.run(self.sample_action_sy, feed_dict={self.states_ph: states})

    def entropy(self, sess, states):
        return sess.run(self.entropy_sy, feed_dict={self.states_ph: states})
