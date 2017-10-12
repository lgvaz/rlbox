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
        # self.config['placeholders']['old_logprob'] = [[None], tf.float32]
        self.placeholders_config['old_logprob'] = [[None], tf.float32]

    def fit(self, sess, states, actions, returns, learning_rate, num_epochs=10, batch_size=64, logger=None):
        # super().fit(self, sess, states, actions, returns, learning_rate, num_epochs=10, batch_size=64, logger=None)

        # Calculate old logprobs
        old_logprob = sess.run(self.policy.logprob_sy, feed_dict={self.placeholders['actions']: actions, self.placeholders['states']: states})
        # Calculate baseline
        if self.use_baseline:
            baseline = self.compute_baseline(sess, states)
        else:
            # Doesn't matter, just used to feed_dict
            baseline = returns

        data = DataGenerator(states, actions, returns, baseline, old_logprob)
        for i_epoch in range(num_epochs):
            data_iterator = data.iterate_once(batch_size)

            for batch in data_iterator:
                states_batch, actions_batch, returns_batch, baseline_batch, old_logprob_batch = batch
                feed_dict = {
                    self.placeholders['states']: states_batch,
                    self.placeholders['actions']: actions_batch,
                    self.placeholders['returns']: returns_batch,
                    self.placeholders['baseline']: baseline_batch,
                    self.placeholders['learning_rate']: learning_rate,
                    self.placeholders['old_logprob']: old_logprob_batch
                }
                loss, _ = sess.run([self.loss_sy, self.training_op], feed_dict=feed_dict)
            logger.add_debug('Loss per batch', loss)

        if logger:
            entropy = self.policy.entropy(sess, states)
            logger.add_log('Learning Rate', learning_rate, precision=5)
            logger.add_log('Entropy', entropy)
