import numpy as np
import tensorflow as tf
from gymmeforce.common.policy import Policy
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import tf_copy_params_op


class PPOModel(VanillaPGModel):
    def __init__(self,
                 env_config,
                 ppo_clip=True,
                 ppo_adaptive_kl=False,
                 kl_coef=1.,
                 kl_targ=0.01,
                 kl_hinge_coef=50.,
                 **kwargs):
        self.ppo_clip = ppo_clip
        self.ppo_adaptive_kl = ppo_adaptive_kl
        self.kl_coef = tf.Variable(kl_coef, name='kl_coef', trainable=False)
        self.kl_targ = kl_targ
        self.kl_hinge_coef = kl_hinge_coef
        self.double_kl_coef_op = self.kl_coef.assign(2 * self.kl_coef)
        self.halve_kl_coef_op = self.kl_coef.assign(0.5 * self.kl_coef)

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
        self._create_surrogate_objective()

    def _set_placeholders_config(self):
        super()._set_placeholders_config()
        self.placeholders_config['ppo_clip_range'] = [[], tf.float32]

    def _create_kl_divergence_op(self):
        self.kl_divergence_sy = self.policy.kl_divergence(self.old_policy, self.policy)

    def _create_old_policy_update_op(self):
        with tf.variable_scope('update_old_policy'):
            self.update_old_policy_op = tf_copy_params_op(
                from_scope='policy', to_scope='old_policy')

    def _update_old_policy(self, sess):
        sess.run(self.update_old_policy_op)

    def _add_losses(self):
        if self.ppo_clip:
            self._clipped_surrogate_loss()
        if self.ppo_adaptive_kl:
            self._kl_loss()
        self._entropy_loss()

        if self.use_baseline:
            self._baseline_loss()

    def _create_surrogate_objective(self):
        with tf.variable_scope('prob_ratio'):
            self.prob_ratio = tf.exp(self.policy.logprob_sy - self.old_policy.logprob_sy)
        with tf.variable_scope('surrogate_objective'):
            self.surrogate = self.prob_ratio * self.placeholders['advantages']

    def _clipped_surrogate_loss(self):
        with tf.variable_scope('clipped_surrogate_objective'):

            clipped_prob_ratio = tf.clip_by_value(
                self.prob_ratio,
                1 - self.placeholders['ppo_clip_range'],
                1 + self.placeholders['ppo_clip_range'],
                name='clipped_prob_ratio')
            clipped_surrogate = clipped_prob_ratio * self.placeholders['advantages']

            clipped_surrogate_losses = tf.minimum(self.surrogate, clipped_surrogate)
            clipped_surrogate_loss = -tf.reduce_mean(clipped_surrogate_losses)

            tf.losses.add_loss(clipped_surrogate_loss)

        # Add logs
        clip_fraction = tf.reduce_mean(
            tf.to_float(
                tf.abs(self.prob_ratio - 1.) > self.placeholders['ppo_clip_range']))
        tf.summary.scalar('policy/ppo_clip_fraction', clip_fraction)
        tf.summary.scalar('policy/ppo_clip_range', self.placeholders['ppo_clip_range'])

    def _kl_loss(self):
        with tf.variable_scope('kl_penalized_surrogate_loss'):
            kl_loss = self.kl_coef * self.kl_divergence_sy
            hinge_loss = self.kl_hinge_coef * tf.maximum(
                0.0, self.kl_divergence_sy - 2 * self.kl_targ)**2

            losses = self.surrogate - kl_loss - hinge_loss
            loss = -tf.reduce_mean(losses)

            tf.losses.add_loss(loss)

        self.callbacks.append(self._kl_callback)

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

    def write_logs(self, sess, logger):
        entropy, kl = sess.run(
            [self.policy.entropy_sy, self.kl_divergence_sy],
            feed_dict=self.placeholders_and_data)
        logger.add_log('policy/Entropy', entropy)
        logger.add_log('policy/KL Divergence', np.mean(kl), precision=4)
        if self.ppo_adaptive_kl:
            logger.add_log('policy/KL Coefficient', sess.run(self.kl_coef))

        self._write_summaries(sess, self.placeholders_and_data)

    def fit(self, sess, *args, **kwargs):
        self._update_old_policy(sess)
        super().fit(sess, *args, **kwargs)
        if self.ppo_adaptive_kl:
            self._update_kl_coef(sess)
