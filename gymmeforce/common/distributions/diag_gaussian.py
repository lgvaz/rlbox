import numpy as np
import tensorflow as tf


class DiagGaussianDist:
    def __init__(self, mean_and_logstd, low_bound=None, high_bound=None):
        self.mean, self.logstd = mean_and_logstd
        self.std = tf.exp(self.logstd)
        self.num_actions = tf.shape(self.mean)[1]
        self.low_bound = low_bound
        self.high_bound = high_bound

    def sample(self):
        sample_action = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        if self.low_bound is not None and self.high_bound is not None:
            sample_action = tf.clip_by_value(sample_action, self.low_bound, self.high_bound)

        return sample_action

    def selected_logprob(self, actions):
        logprob = -0.5 * tf.log(2 * np.pi)
        logprob += -0.5 * tf.reduce_sum(self.logstd)
        logprob += -0.5 * tf.reduce_sum(((actions - self.mean) / self.std) ** 2, axis=1)

        return logprob

    def entropy(self):
        entropy = 0.5 * (tf.reduce_sum(self.logstd)
                         + tf.to_float(self.num_actions) * (tf.log(2 * np.pi * np.e)))
        return entropy

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        '''
        From https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Examples
        The terms are in order with the page
        '''
        assert isinstance(old_dist, DiagGaussianDist) and isinstance(new_dist, DiagGaussianDist)
        first_term = tf.reduce_sum(tf.exp(old_dist.logstd - new_dist.logstd))
        second_term = tf.reduce_sum(((new_dist.mean - old_dist.mean) ** 2)
                                    / new_dist.std, axis=1)
        third_term = tf.to_float(new_dist.num_actions)
        fourth_term = tf.reduce_sum(new_dist.logstd) - tf.reduce_sum(old_dist.logstd)

        kl = 0.5 * (first_term + second_term - third_term + fourth_term)
        kl = tf.reduce_mean(kl)

        return kl
