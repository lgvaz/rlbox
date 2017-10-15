import tensorflow as tf
from gymmeforce.models import VanillaPGModel


class PPOModel(VanillaPGModel):
    def __init__(self, env_config, epsilon_clip=0.2, **kwargs):
        self.epsilon_clip = epsilon_clip
        super().__init__(env_config, **kwargs)

    def _add_losses(self):
        self._clipped_surrogate_loss(self.policy, self.epsilon_clip)
        self._entropy_loss(self.policy, self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.baseline_target)

    def _clipped_surrogate_loss(self, policy, epsilon_clip):
        advantages = self._estimate_advatanges()
        prob_ratio = tf.exp(policy.logprob_sy - self.placeholders['old_logprob'])
        clipped_prob_ratio = tf.clip_by_value(prob_ratio,
                                              1 - epsilon_clip,
                                              1 + epsilon_clip)
        clipped_surrogate_losses = tf.minimum(prob_ratio * advantages,
                                              clipped_prob_ratio * advantages)
        clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

        tf.losses.add_loss(clipped_surrogate_loss)

    def _set_placeholders_config(self):
        super()._set_placeholders_config()
        self.placeholders_config['old_logprob'] = [[None], tf.float32]

    def _fetch_placeholders_data_dict(self, sess, states, actions, returns):
        super()._fetch_placeholders_data_dict(sess, states, actions, returns)
        # Add old_logprob to feed_dict fetching
        feed_dict = {
            self.placeholders['states']: states,
            self.placeholders['actions']: actions
        }
        old_logprob = sess.run(self.policy.logprob_sy, feed_dict=feed_dict)
        self.placeholders_and_data[self.placeholders['old_logprob']] = old_logprob


    def fit(self, sess, states, actions, returns, learning_rate, **kwargs):
        super().fit(sess, states, actions, returns, learning_rate, **kwargs)
