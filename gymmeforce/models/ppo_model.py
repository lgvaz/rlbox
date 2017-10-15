import tensorflow as tf
from gymmeforce.common.policy import Policy
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import tf_copy_params_op


class PPOModel(VanillaPGModel):
    def __init__(self, env_config, epsilon_clip=0.2, **kwargs):
        self.epsilon_clip = epsilon_clip
        super().__init__(env_config, **kwargs)
        self.update_old_policy_op = tf_copy_params_op(from_scope='policy',
                                                      to_scope='old_policy')
        self.kl_divergence_sy = self.policy.kl_divergence(self.old_policy, self.policy)

    def _create_policy(self, states_ph, actions_ph, policy_graph):
        self.policy = Policy(self.env_config, states_ph, actions_ph, policy_graph,
                             scope='policy')
        self.old_policy = Policy(self.env_config, states_ph, actions_ph, policy_graph,
                                 scope='old_policy', trainable=False)

    def _add_losses(self):
        self._clipped_surrogate_loss(self.policy, self.epsilon_clip)
        self._entropy_loss(self.policy, self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.baseline_target)

    def _clipped_surrogate_loss(self, policy, epsilon_clip):
        advantages = self._estimate_advatanges()
        prob_ratio = tf.exp(policy.logprob_sy - self.old_policy.logprob_sy)
        clipped_prob_ratio = tf.clip_by_value(prob_ratio,
                                              1 - epsilon_clip,
                                              1 + epsilon_clip)
        clipped_surrogate_losses = tf.minimum(prob_ratio * advantages,
                                              clipped_prob_ratio * advantages)
        clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

        tf.losses.add_loss(clipped_surrogate_loss)

    def _update_old_policy(self, sess):
        sess.run(self.update_old_policy_op)

    def write_logs(self, sess, logger):
        self.write_summaries(sess, self.placeholders_and_data)

        entropy, kl = sess.run([self.policy.entropy_sy, self.kl_divergence_sy],
                               feed_dict=self.placeholders_and_data)
        logger.add_log('Entropy', entropy)
        logger.add_log('KL Divergence', kl, precision=4)

    def fit(self, sess, states, actions, returns, learning_rate, **kwargs):
        self._update_old_policy(sess)
        super().fit(sess, states, actions, returns, learning_rate, **kwargs)
