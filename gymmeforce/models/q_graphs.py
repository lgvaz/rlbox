import tensorflow as tf

def deepmind_graph(states, num_actions, scope, dueling=False, reuse=None):
    ''' Network graph from DeepMind '''
    with tf.variable_scope(scope, reuse=reuse):
        # graph architecture
        net = states
        # Convolutional layers
        with tf.variable_scope('convolutions', reuse=reuse):
            net = tf.layers.conv2d(net, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu)
            net = tf.contrib.layers.flatten(net)

        if dueling:
            with tf.variable_scope('dueling_net', reuse=reuse):
                with tf.variable_scope('state_value', reuse=reuse):
                    state_value = tf.layers.dense(net, 512, activation=tf.nn.relu)
                    state_value = tf.layers.dense(state_value, 1)

                with tf.variable_scope('advantages', reuse=reuse):
                    advantages = tf.layers.dense(net, 512, activation=tf.nn.relu)
                    advantages = tf.layers.dense(advantages, num_actions)
                    advantages_mean = tf.reduce_mean(advantages, 1, keep_dims=True)

                q_values = state_value + (advantages - advantages_mean)

        else:
            with tf.variable_scope('q_values', reuse=reuse):
                net = tf.layers.dense(net, 512, activation=tf.nn.relu)
                q_values = tf.layers.dense(net, num_actions, name='Q_{}'.format(scope))

        return q_values

def simple_graph(states, num_actions, scope, dueling=False, reuse=None):
    ''' Simple fully connected graph '''
    with tf.variable_scope(scope, reuse=reuse):
        # graph architecture
        net = states
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)

        if dueling:
            with tf.variable_scope('dueling_net', reuse=reuse):
                with tf.variable_scope('state_value', reuse=reuse):
                    state_value = tf.layers.dense(net, 64, activation=tf.nn.relu)
                    state_value = tf.layers.dense(state_value, 1)

                with tf.variable_scope('advantages', reuse=reuse):
                    advantages = tf.layers.dense(net, 64, activation=tf.nn.relu)
                    advantages = tf.layers.dense(advantages, num_actions)
                    advantages_mean = tf.reduce_mean(advantages, 1, keep_dims=True)

                q_values = state_value + (advantages - advantages_mean)

        else:
            with tf.variable_scope('q_values', reuse=reuse):
                net = tf.layers.dense(net, 64, activation=tf.nn.relu)
                q_values = tf.layers.dense(net, num_actions, name='Q_{}'.format(scope))

        return q_values
