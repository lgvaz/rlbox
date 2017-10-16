import tensorflow as tf
from gymmeforce.common.policy import Policy
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import tf_copy_params_op


class PPOModel(VanillaPGModel):
    def __init__(self, env_config, epsilon_clip=0.2, **kwargs):
        self.epsilon_clip = epsilon_clip
        super().__init__(env_config, **kwargs)

    def _create_policy(self, states_ph, actions_ph, policy_graph):
        self.policy = Policy(self.env_config, states_ph, actions_ph, policy_graph,
                             scope='policy')
        self.old_policy = Policy(self.env_config, states_ph, actions_ph, policy_graph,
                                 scope='old_policy', trainable=False)
        self._create_old_policy_update_op()
        self._create_kl_divergence_op()

    def _add_losses(self):
        self._clipped_surrogate_loss(self.policy, self.epsilon_clip)
        self._entropy_loss(self.policy, self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.baseline_target)

    def _clipped_surrogate_loss(self, policy, epsilon_clip):
        with tf.variable_scope('L_clip'):
            with tf.variable_scope('prob_ratio'):
                prob_ratio = tf.exp(policy.logprob_sy - self.old_policy.logprob_sy)
                # TODO: I don't think is correct to take mean here
                # prob_ratio = tf.reduce_mean(prob_ratio, axis=1)
                # prob_ratio = tf.exp(policy.logprob_sy - self.placeholders['old_logprob'])
            clipped_prob_ratio = tf.clip_by_value(prob_ratio,
                                                  1 - epsilon_clip,
                                                  1 + epsilon_clip,
                                                  name='clipped_prob_ratio')
            with tf.variable_scope('clipped_surrogate_loss'):
                with tf.variable_scope('surrogate_objective'):
                    surrogate = prob_ratio * self.advantages
                with tf.variable_scope('clipped_surrogate_objective'):
                    clipped_surrogate = clipped_prob_ratio * self.advantages
                clipped_surrogate_losses = tf.minimum(surrogate, clipped_surrogate)
                clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

            tf.losses.add_loss(clipped_surrogate_loss)

    def _create_kl_divergence_op(self):
        self.kl_divergence_sy = self.policy.kl_divergence(self.old_policy, self.policy)

    def _create_old_policy_update_op(self):
        with tf.variable_scope('update_old_policy'):
            self.update_old_policy_op = tf_copy_params_op(from_scope='policy',
                                                          to_scope='old_policy')
    def _update_old_policy(self, sess):
        sess.run(self.update_old_policy_op)

    # def _set_placeholders_config(self):
    #     super()._set_placeholders_config()
    #     self.placeholders_config['old_logprob'] = [[None], tf.float32]

    # def _fetch_placeholders_data_dict(self, sess, states, actions, returns):
    #     super()._fetch_placeholders_data_dict(sess, states, actions, returns)
    #     # Add old_logprob to feed_dict fetching
    #     old_logprob = sess.run(self.policy.logprob_sy, feed_dict=self.placeholders_and_data)
    #     self.placeholders_and_data[self.placeholders['old_logprob']] = old_logprob

    def write_logs(self, sess, logger):
        entropy, kl = sess.run([self.policy.entropy_sy,
                                self.kl_divergence_sy],
                               feed_dict=self.placeholders_and_data)

        logger.add_log('policy/Entropy', entropy)
        logger.add_log('policy/KL Divergence', kl, precision=4)

        self.write_summaries(sess, self.placeholders_and_data)

    def fit(self, sess, states, actions, returns, learning_rate, **kwargs):
        self._update_old_policy(sess)
        super().fit(sess, states, actions, returns, learning_rate, **kwargs)
