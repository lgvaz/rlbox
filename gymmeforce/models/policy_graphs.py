import tensorflow as tf

def dense_policy_graph(inputs, env_config, activation_fn=tf.nn.tanh,
                       scope='policy_graph', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        net = tf.contrib.layers.flatten(net)
        # TODO: Custom weights initializer
        # net = tf.layers.dense(net, 256, activation=activation_fn)
        net = tf.layers.dense(net, 64, activation=activation_fn)
        net = tf.layers.dense(net, 64, activation=activation_fn)

        if env_config['action_space'] == 'continuous':
            mean = tf.layers.dense(net, env_config['num_actions'], name='mean')
            logstd = tf.get_variable('logstd', (env_config['num_actions'],), tf.float32)
            return mean, logstd
        if env_config['action_space'] == 'discrete':
            logits = tf.layers.dense(net, env_config['num_actions'], name='logits')
            return logits
