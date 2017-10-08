import tensorflow as tf

class CategoricalDist:
    def __init__(self, logits):
        self.logits = logits
        self.num_actions = tf.shape(logits)[1]

    def sample(self, num_samples=1):
        return tf.squeeze(tf.multinomial(self.logits, num_samples))

    def logprob(self, actions):
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        logprob = tf.nn.log_softmax(self.logits)
        logprob = tf.reduce_sum(one_hot_actions * logprob, axis=1)
        return logprob
