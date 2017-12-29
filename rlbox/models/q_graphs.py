import numpy as np
import tensorflow as tf


def deepmind_graph(states, output_size, scope, dueling=False, reuse=None):
    ''' Network graph from DeepMind '''
    # if tf.uint8 == self.env_config['input_type']:
    # Convert to float on GPU
    #     states_t = tf.cast(self.placeholders['states_t'], tf.float32) / 255.
    #     states_tp1 = tf.cast(self.placeholders['states_tp1'], tf.float32) / 255.
    # else:
    #     states_t = self.placeholders['states_t']
    #     states_tp1 = self.placeholders['states_tp1']

    with tf.variable_scope(scope, reuse=reuse):
        # Converted to float
        net = tf.to_float(states) / 255
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
                    advantages = tf.layers.dense(advantages, output_size)
                    advantages_mean = tf.reduce_mean(advantages, 1, keep_dims=True)

                q_values = state_value + (advantages - advantages_mean)

        else:
            with tf.variable_scope('q_values', reuse=reuse):
                net = tf.layers.dense(net, 512, activation=tf.nn.relu)
                q_values = tf.layers.dense(
                    net, np.prod(output_size), name='Q_{}'.format(scope))

        return tf.reshape(q_values, output_size)


def simple_graph(states, output_size, scope, dueling=False, reuse=None):
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
                    advantages = tf.layers.dense(advantages, output_size)
                    advantages_mean = tf.reduce_mean(advantages, 1, keep_dims=True)

                q_values = state_value + (advantages - advantages_mean)

        else:
            with tf.variable_scope('q_values', reuse=reuse):
                net = tf.layers.dense(net, 64, activation=tf.nn.relu)
                q_values = tf.layers.dense(
                    net, np.prod(output_size), name='Q_{}'.format(scope))

        return tf.reshape(q_values, (-1, ) + output_size)
        # return q_values
