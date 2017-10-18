import numpy as np
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

    def _create_kl_divergence_op(self):
        self.kl_divergence_sy = self.policy.kl_divergence(self.old_policy, self.policy)

    def _create_old_policy_update_op(self):
        with tf.variable_scope('update_old_policy'):
            self.update_old_policy_op = tf_copy_params_op(from_scope='policy',
                                                          to_scope='old_policy')

    def _add_losses(self):
        self._clipped_surrogate_loss(self.policy, self.epsilon_clip)
        self._entropy_loss(self.policy, self.entropy_coef)
        if self.use_baseline:
            self._baseline_loss(self.value_fn_sy, self.value_fn_target)

    def _clipped_surrogate_loss(self, policy, epsilon_clip):
        with tf.variable_scope('L_clip'):
            with tf.variable_scope('prob_ratio'):
                self.prob_ratio = tf.exp(policy.logprob_sy - self.old_policy.logprob_sy)
                # Alternatively we can use a placeholder to feed the old logprobs
                # self.prob_ratio = tf.exp(policy.logprob_sy - self.placeholders['old_logprob'])
            self.clipped_prob_ratio = tf.clip_by_value(self.prob_ratio,
                                                       1 - epsilon_clip,
                                                       1 + epsilon_clip,
                                                       name='clipped_prob_ratio')
            with tf.variable_scope('clipped_surrogate_loss'):
                with tf.variable_scope('surrogate_objective'):
                    surrogate = self.prob_ratio * self.placeholders['advantages']
                with tf.variable_scope('clipped_surrogate_objective'):
                    clipped_surrogate = self.clipped_prob_ratio * self.placeholders['advantages']
                clipped_surrogate_losses = tf.minimum(surrogate, clipped_surrogate)
                clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

            tf.losses.add_loss(clipped_surrogate_loss)

    def _update_old_policy(self, sess):
        sess.run(self.update_old_policy_op)

    # This lines show an example of how to add an additional placeholder which
    # will be fetched during the training operation
    # def _set_placeholders_config(self):
    #     super()._set_placeholders_config()
    #     self.placeholders_config['old_logprob'] = [[None], tf.float32]

    # def _fetch_placeholders_data_dict(self, sess, states, actions, returns):
    #     super()._fetch_placeholders_data_dict(sess, states, actions, returns)
    #     # Add old_logprob to feed_dict fetching
    #     old_logprob = sess.run(self.policy.logprob_sy,
    #                            feed_dict=self.placeholders_and_data)
    #     self.placeholders_and_data[self.placeholders['old_logprob']] = old_logprob

    def _create_summaries_op(self):
        super()._create_summaries_op()

        tf.summary.histogram('policy/prob_ratio', self.prob_ratio)
        tf.summary.scalar('policy/prob_ratio/mean',
                          tf.reduce_mean(self.prob_ratio))
        tf.summary.scalar('policy/prob_ratio/max',
                          tf.reduce_max(self.prob_ratio))
        tf.summary.scalar('policy/prob_ratio/min',
                          tf.reduce_min(self.prob_ratio))

        tf.summary.histogram('policy/clipped_prob_ratio', self.clipped_prob_ratio)
        tf.summary.scalar('policy/clipped_prob_ratio/mean',
                          tf.reduce_mean(self.clipped_prob_ratio))
        tf.summary.scalar('policy/clipped_prob_ratio/max',
                          tf.reduce_max(self.clipped_prob_ratio))
        tf.summary.scalar('policy/clipped_prob_ratio/min',
                          tf.reduce_min(self.clipped_prob_ratio))

    def write_logs(self, sess, logger):
        entropy, kl = sess.run([self.policy.entropy_sy,
                                self.kl_divergence_sy],
                               feed_dict=self.placeholders_and_data)
        logger.add_log('policy/Entropy', entropy)
        logger.add_log('policy/KL Divergence', np.mean(kl), precision=4)

        if self.use_baseline:
            y_pred = sess.run(self.baseline_sy, feed_dict=self.placeholders_and_data)
            y_true = self.placeholders_and_data[self.placeholders['returns']]
            explained_variance = np.var(y_true - y_pred) / np.var(y_true)
            logger.add_log('baseline/Explained Variance', explained_variance)

        self.write_summaries(sess, self.placeholders_and_data)

    def fit(self, sess, states, actions, returns, learning_rate, **kwargs):
        self._update_old_policy(sess)
        super().fit(sess, states, actions, returns, learning_rate, **kwargs)
