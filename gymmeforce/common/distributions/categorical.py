import tensorflow as tf

class CategoricalDist:
    def __init__(self, logits):
        self.logits = logits
        self.num_actions = tf.shape(logits)[1]
        self.logprob = tf.nn.log_softmax(logits)

    def sample(self, num_samples=1):
        return tf.squeeze(tf.multinomial(self.logits, num_samples))

    def selected_logprob(self, actions):
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        selected_logprob = tf.reduce_sum(one_hot_actions * self.logprob, axis=1)

        return selected_logprob

    def entropy(self):
        return -(tf.reduce_sum(tf.exp(self.logprob) * self.logprob)
                 / tf.to_float(tf.shape(self.logprob)[0]))
