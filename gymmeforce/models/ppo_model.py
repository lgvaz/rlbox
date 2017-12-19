import numpy as np
import tensorflow as tf
from gymmeforce.common.policy import Policy
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import tf_copy_params_op


class PPOModel(VanillaPGModel):
    def __init__(self,
                 env_config,
                 epsilon_clip=0.2,
                 kl_coef=1.,
                 kl_targ=0.01,
                 kl_hinge_coef=50.,
                 **kwargs):
        self.epsilon_clip = epsilon_clip
        self.kl_coef = tf.Variable(kl_coef, name='kl_coef', trainable=False)
        self.double_kl_coef_op = self.kl_coef.assign(2 * self.kl_coef)
        self.halve_kl_coef_op = self.kl_coef.assign(0.5 * self.kl_coef)
        self.kl_targ = 0.01
        self.kl_hinge_coef = 50
        super().__init__(env_config, **kwargs)

    def _create_policy(self):
        super()._create_policy()
        self.old_policy = Policy(
            env_config=self.env_config,
            states_ph=self.placeholders['states'],
            actions_ph=self.placeholders['actions'],
            graph=self.policy_graph,
            scope='old_policy',
            trainable=False)

        self._create_old_policy_update_op()
        self._create_kl_divergence_op()

    def _create_kl_divergence_op(self):
        self.kl_divergence_sy = self.policy.kl_divergence(self.old_policy, self.policy)

    def _create_old_policy_update_op(self):
        with tf.variable_scope('update_old_policy'):
            self.update_old_policy_op = tf_copy_params_op(
                from_scope='policy', to_scope='old_policy')

    def _update_old_policy(self, sess):
        sess.run(self.update_old_policy_op)

    def _add_losses(self):
        # self._clipped_surrogate_loss()
        self._kl_loss()
        self._entropy_loss()
        if self.use_baseline:
            self._baseline_loss()

    def _clipped_surrogate_loss(self):
        with tf.variable_scope('L_clip'):
            with tf.variable_scope('prob_ratio'):
                self.prob_ratio = tf.exp(self.policy.logprob_sy -
                                         self.old_policy.logprob_sy)

            self.clipped_prob_ratio = tf.clip_by_value(
                self.prob_ratio,
                1 - self.epsilon_clip,
                1 + self.epsilon_clip,
                name='clipped_prob_ratio')

            with tf.variable_scope('clipped_surrogate_loss'):
                with tf.variable_scope('surrogate_objective'):
                    surrogate = self.prob_ratio * self.placeholders['advantages']
                with tf.variable_scope('clipped_surrogate_objective'):
                    clipped_surrogate = self.clipped_prob_ratio * self.placeholders[
                        'advantages']
                clipped_surrogate_losses = tf.minimum(surrogate, clipped_surrogate)
                clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

            tf.losses.add_loss(clipped_surrogate_loss)

    def _kl_loss(self):
        prob_ratio = tf.exp(self.policy.logprob_sy - self.old_policy.logprob_sy)
        surrogate = prob_ratio * self.placeholders['advantages']

        kl_loss = self.kl_coef * self.kl_divergence_sy

        hinge_loss = 50 * tf.maximum(0.0, self.kl_divergence_sy - 2 * self.kl_targ)**2

        # losses = -(tf.reduce_mean(surrogate) - tf.reduce_mean(kl_loss))
        with tf.variable_scope('kl_penalized_surrogate_loss'):
            losses = surrogate - kl_loss - hinge_loss
            loss = -tf.reduce_mean(losses)

        tf.losses.add_loss(loss)

        tf.summary.scalar('losses/kl_loss/mean', tf.reduce_mean(kl_loss))
        tf.summary.scalar('losses/surrogate/mean', tf.reduce_mean(surrogate))

    def _update_kl_coef(self, sess):
        kl = np.mean(
            sess.run(self.kl_divergence_sy, feed_dict=self.placeholders_and_data))

        if kl < self.kl_targ / 1.5:
            sess.run(self.halve_kl_coef_op)
        if kl > self.kl_targ * 1.5:
            sess.run(self.double_kl_coef_op)

    def _kl_callback(self, sess):
        kl = np.mean(
            sess.run(self.kl_divergence_sy, feed_dict=self.placeholders_and_data))

        if kl > 4 * self.kl_targ:
            print('*** KL divergence too high, early stopping ***')
            return True

    def _create_summaries_op(self):
        super()._create_summaries_op()

        # tf.summary.histogram('policy/prob_ratio', self.prob_ratio)
        # tf.summary.scalar('policy/prob_ratio/mean', tf.reduce_mean(self.prob_ratio))
        # tf.summary.scalar('policy/prob_ratio/max', tf.reduce_max(self.prob_ratio))
        # tf.summary.scalar('policy/prob_ratio/min', tf.reduce_min(self.prob_ratio))

        # tf.summary.histogram('policy/clipped_prob_ratio', self.clipped_prob_ratio)
        # tf.summary.scalar('policy/clipped_prob_ratio/mean',
        #                   tf.reduce_mean(self.clipped_prob_ratio))
        # tf.summary.scalar('policy/clipped_prob_ratio/max',
        #                   tf.reduce_max(self.clipped_prob_ratio))
        # tf.summary.scalar('policy/clipped_prob_ratio/min',
        #                   tf.reduce_min(self.clipped_prob_ratio))

    def write_logs(self, sess, logger):
        entropy, kl = sess.run(
            [self.policy.entropy_sy, self.kl_divergence_sy],
            feed_dict=self.placeholders_and_data)
        logger.add_log('policy/Entropy', entropy)
        logger.add_log('policy/KL Divergence', np.mean(kl), precision=4)
        logger.add_log('policy/KL Coefficient', sess.run(self.kl_coef))

        self._write_summaries(sess, self.placeholders_and_data)

    def fit(self, sess, *args, **kwargs):
        self._update_old_policy(sess)
        super().fit(sess, *args, callbacks=[self._kl_callback], **kwargs)
        self._update_kl_coef(sess)
