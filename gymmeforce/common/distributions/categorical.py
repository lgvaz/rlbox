import tensorflow as tf


class CategoricalDist:
    def __init__(self, logits):
        self.logits = logits
        self.num_actions = tf.shape(logits)[1]
        self.logprob_sy = tf.nn.log_softmax(logits)
        self.action_probs_sy = tf.nn.softmax(logits)

    def sample(self, num_samples=1):
        return tf.squeeze(tf.multinomial(self.logits, num_samples))

    def selected_logprob(self, actions):
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        selected_logprob = tf.reduce_sum(one_hot_actions * self.logprob_sy, axis=1)

        return selected_logprob

    def entropy(self):
        return -tf.reduce_mean(tf.reduce_sum(tf.exp(self.logprob_sy) * self.logprob_sy, axis=1))

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        assert isinstance(old_dist, CategoricalDist) and isinstance(new_dist, CategoricalDist)
        kl_sy = tf.reduce_sum(old_dist.action_probs_sy
                              * (old_dist.logprob_sy - new_dist.logprob_sy), axis=1)
        kl_sy = tf.reduce_mean(kl_sy)

        return kl_sy
