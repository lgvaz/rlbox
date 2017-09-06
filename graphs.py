import tensorflow as tf

def deepmind_graph(states, num_actions, scope, reuse=None):
    ''' Network graph from DeepMind '''
    with tf.variable_scope(scope, reuse=reuse):
        # graph architecture
        net = states
        # Convolutional layers
        net = tf.layers.conv2d(net, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 32, (3, 3), strides=(1, 1), activation=tf.nn.relu)
        net = tf.contrib.layers.flatten(net)

        # Dense layers
        net = tf.layers.dense(net, 512, activation=tf.nn.relu)
        output = tf.layers.dense(net, num_actions, name='Q_{}'.format(scope))

        return output

def simple_graph(states, num_actions, scope, reuse=None):
    ''' Simple fully connected graph '''
    with tf.variable_scope(scope, reuse=reuse):
        # graph architecture
        net = states
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 512, activation=tf.nn.relu)
        output = tf.layers.dense(net, num_actions, name='Q_{}'.format(scope))

        return output
