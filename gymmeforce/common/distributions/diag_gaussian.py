import numpy as np
import tensorflow as tf

class DiagGaussianDist:
    def __init__(self, mean, logstd, low_bound=None, high_bound=None):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        self.num_actions = tf.shape(mean)[1]
        self.low_bound = low_bound
        self.high_bound = high_bound

    def sample(self):
        sample_action = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        # TODO: Clipping actions is making logstd to explode!! DONT THINK SO
        if self.low_bound is not None and self.high_bound is not None:
            sample_action = tf.clip_by_value(sample_action, self.low_bound, self.high_bound)

        return sample_action

    def logprob(self, actions):
        # TODO: problem not here
        logprob = -0.5 * tf.log(2 * np.pi)
        logprob += -0.5 * tf.reduce_sum(self.logstd)
        logprob += -0.5 * tf.reduce_sum(((actions - self.mean) / self.std) ** 2, axis=1)

        return logprob

    def entropy(self):
        entropy = 0.5 * (tf.reduce_sum(self.logstd)
                         + tf.to_float(self.num_actions) * (tf.log(2 * np.pi * np.e)))
        # entropy = tf.reduce_sum(self.logstd + 0.5 * tf.log(2 * np.pi * np.e), axis=1)
        # entropy = 0.5 * tf.reduce_sum(self.log)
        return entropy
