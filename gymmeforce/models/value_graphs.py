import tensorflow as tf

def dense_value_graph(inputs, activation_fn=tf.nn.tanh, scope='value_graphs', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        net = tf.contrib.layers.flatten(net)
        # net = tf.layers.dense(net, 256, activation=activation_fn)
        # net = tf.layers.dense(net, 128, activation=activation_fn)
        net = tf.layers.dense(net, 64, activation=activation_fn)
        net = tf.layers.dense(net, 64, activation=activation_fn)
        state_value = tf.layers.dense(net, 1)

        return tf.squeeze(state_value)
