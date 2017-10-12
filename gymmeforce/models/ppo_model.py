import tensorflow as tf
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.data_gen import DataGenerator


class PPOModel(VanillaPGModel):
    def __init__(self, env_config, normalize_advantages=True, use_baseline=True, normalize_baseline=True, entropy_coef=0., policy_graph=None, value_graph=None, input_type=None, log_dir=None):
        super().__init__(env_config, normalize_advantages, use_baseline, normalize_baseline, entropy_coef, policy_graph, value_graph, input_type, log_dir)


    def _add_losses(self):
        self._surrogate_loss()
        if self.use_baseline:
            self._baseline_loss(self.baseline_sy, self.baseline_target)

    def _surrogate_loss(self):
        advantages = self._estimate_advatanges()
        prob_ratio = tf.exp(self.policy.logprob_sy - self.placeholders['old_logprob'])
        clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - 0.2, 1 + 0.2)
        surrogate_losses = tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)
        surrogate_loss = -tf.reduce_mean(surrogate_losses)

        tf.losses.add_loss(surrogate_loss)

    def _set_placeholders_config(self):
        super()._set_placeholders_config()
        self.placeholders_config['old_logprob'] = [[None], tf.float32]

    def _fetch_placeholders_data_dict(self, sess, states, actions, returns):
        super()._fetch_placeholders_data_dict(sess, states, actions, returns)

        # Calculate old logprobs
        old_logprob = sess.run(self.policy.logprob_sy, feed_dict={self.placeholders['actions']: actions, self.placeholders['states']: states})
        self.placeholders_and_data[self.placeholders['old_logprob']] = old_logprob


    def fit(self, sess, states, actions, returns, learning_rate, num_epochs=10, batch_size=64, logger=None):
        super().fit(sess, states, actions, returns, learning_rate, num_epochs, batch_size, logger)

