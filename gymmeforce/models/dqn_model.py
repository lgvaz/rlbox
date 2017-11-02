import numpy as np
import tensorflow as tf
from gymmeforce.models.q_graphs import deepmind_graph, simple_graph
from gymmeforce.models.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, env_config, graph=None, double=False, dueling=False, **kwargs):
        super().__init__(env_config, **kwargs)
        self.double = double
        self.dueling= dueling

        # If input is an image defaults to deepmind_graph, else simple_graph
        if graph is None:
            if len(env_config['state_shape']) == 3:
                self.graph = deepmind_graph
                print('Using deepmind_graph')
            else:
                self.graph = simple_graph
                print('Using simple_graph')
        else:
            self.graph = graph
            print('Using custom graph')

        self._set_placeholders_config()
        self._create_placeholders(self.placeholders_config)

        self._create_graphs()

        # Training ops
        self.training_op = None
        self.update_target_op = None

        # Create collections for loading later
        tf.add_to_collection('state_input', self.placeholders['states_t'])
        tf.add_to_collection('q_online_t', self.q_online_t)

    def _set_placeholders_config(self):
        self.placeholders_config = {
            'states_t': [[None] + list(self.env_config['state_shape']),
                         self.env_config['input_type']],
            'states_tp1': [[None] + list(self.env_config['state_shape']),
                           self.env_config['input_type']],
            'actions': [[None], tf.int32],
            'rewards': [[None], tf.float32],
            'dones': [[None], tf.float32],
            'learning_rate': [[], tf.float32]
        }

    def _create_graphs(self):
        #TODO: FIX THIS
        if tf.uint8 == self.env_config['input_type']:
            # Convert to float on GPU
            states_t = tf.cast(self.placeholders['states_t'], tf.float32) / 255.
            states_tp1 = tf.cast(self.placeholders['states_tp1'], tf.float32) / 255.
        else:
            states_t = self.placeholders['states_t']
            states_tp1 = self.placeholders['states_tp1']

        self.q_online_t = self.graph(states_t,
                                     self.env_config['num_actions'],
                                     'online',
                                     dueling=self.dueling)
        self.q_target_tp1 = self.graph(states_tp1,
                                       self.env_config['num_actions'],
                                       'target',
                                       dueling=self.dueling)
        if self.double:
            self.q_online_tp1 = self.graph(states_tp1,
                                           self.env_config['num_actions'],
                                           'online',
                                           dueling=self.dueling,
                                           reuse=True)

    def _build_optimization(self, clip_norm, gamma, n_step):
        # Choose only the q values for selected actions
        onehot_actions = tf.one_hot(self.placeholders['actions'],
                                    self.env_config['num_actions'])
        q_t = tf.reduce_sum(tf.multiply(self.q_online_t, onehot_actions), axis=1)

        # Caculate td_target
        if self.double:
            best_actions_onehot = tf.one_hot(tf.argmax(self.q_online_tp1, axis=1), self.env_config['num_actions'])
            q_tp1 = tf.reduce_sum(tf.multiply(self.q_target_tp1, best_actions_onehot), axis=1)
        else:
            q_tp1 = tf.reduce_max(self.q_target_tp1, axis=1)
        td_target = (self.placeholders['rewards']
                     + (1 - self.placeholders['dones']) * (gamma ** n_step) * q_tp1)
        errors = tf.losses.huber_loss(labels=td_target, predictions=q_t)
        self.total_error = tf.reduce_mean(errors)

        # Create training operation
        opt = tf.train.AdamOptimizer(self.placeholders['learning_rate'], epsilon=1e-4)
        # Clip gradients
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online')
        grads_and_vars = opt.compute_gradients(self.total_error, online_vars)
        with tf.variable_scope('gradient_clipping'):
            clipped_grads = [(tf.clip_by_norm(grad, clip_norm), var)
                            for grad, var in grads_and_vars if grad is not None]
        training_op = opt.apply_gradients(clipped_grads)

        return training_op

    def _build_target_update_op(self, target_soft_update=1.):
        # Get variables within defined scope
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'online')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
        # Create operations that copy the variables
        op_holder = [target_var.assign(target_soft_update * online_var
                                       + (1 - target_soft_update) * target_var)
                     for online_var, target_var in zip(online_vars, target_vars)]

        return op_holder

    def _create_summaries_op(self):
        self._maybe_create_writer()
        tf.summary.scalar('network/loss', self.total_error)
        tf.summary.scalar('network/Q_mean', tf.reduce_mean(self.q_online_t))
        tf.summary.scalar('network/Q_max', tf.reduce_max(self.q_online_t))
        tf.summary.histogram('network/q_values', self.q_online_t)

    def create_training_ops(self, gamma, clip_norm, n_step, target_soft_update):
        # Create training operations
        self.training_op = self._build_optimization(clip_norm, gamma, n_step)
        self.update_target_op = self._build_target_update_op(target_soft_update)

    def predict(self, sess, states):
        return sess.run(self.q_online_t, feed_dict={self.placeholders['states_t']: states})

    def target_predict(self, sess, states):
        return sess.run(self.q_target_tp1, feed_dict={self.placeholders['states_tp1']: states})

    def update_target_net(self, sess):
        sess.run(self.update_target_op)

    def fit(self, sess, learning_rate, states_t, states_tp1, actions, rewards, dones):
        feed_dict = {
            self.placeholders['states_t']: states_t,
            self.placeholders['states_tp1']: states_tp1,
            self.placeholders['actions']: actions,
            self.placeholders['rewards']: rewards,
            self.placeholders['dones']: dones,
            self.placeholders['learning_rate']: learning_rate
        }
        sess.run(self.training_op, feed_dict=feed_dict)

    def write_summaries(self, sess, step, states_t, states_tp1, actions, rewards, dones):
        if self.merged is None:
            self._create_summaries_op()
            self.merged = tf.summary.merge_all()

        feed_dict = {
            self.placeholders['states_t']: states_t,
            self.placeholders['states_tp1']: states_tp1,
            self.placeholders['actions']: actions,
            self.placeholders['rewards']: rewards,
            self.placeholders['dones']: dones
        }
        summary = sess.run(self.merged, feed_dict=feed_dict)
        self._writer.add_summary(summary, global_step=step)
