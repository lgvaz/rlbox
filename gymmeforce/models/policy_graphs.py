import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

def dense_policy_graph(inputs, env_config, activation_fn=tf.nn.tanh,
                       scope='policy_graph', reuse=None, trainable=True):
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        net = tf.contrib.layers.flatten(net)
        # net = tf.layers.dense(net, 128, activation=activation_fn,
                              # kernel_initializer=variance_scaling_initializer(factor=1))
        net = tf.layers.dense(net, 64, activation=activation_fn,
                              kernel_initializer=variance_scaling_initializer(factor=1),
                              trainable=trainable)
        net = tf.layers.dense(net, 64, activation=activation_fn,
                              kernel_initializer=variance_scaling_initializer(factor=1),
                              trainable=trainable)

        if env_config['action_space'] == 'continuous':
            mean = tf.layers.dense(net, env_config['num_actions'],
                                   kernel_initializer=variance_scaling_initializer(factor=1),
                                   name='mean',
                                   trainable=trainable)
            logstd = tf.get_variable('logstd', (1, env_config['num_actions']), tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=trainable)
            # Clip logstd, because it likes to go to NaN
            logstd = tf.clip_by_value(logstd, np.log(1e-7), -np.log(1e-7))
            return mean, logstd
        if env_config['action_space'] == 'discrete':
            logits = tf.layers.dense(net, env_config['num_actions'],
                                     name='logits',
                                     trainable=trainable)
            return logits
