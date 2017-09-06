import numpy as np
import tensorflow as tf

class BaseAgent:
    def __init__(self, env, log_dir):
        self.env = env
        self.state = env.reset()
        self.log_dir = log_dir
        self.sess = None

    def _maybe_create_tf_sess(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _play_one_step(self, epsilon, render):
        if render:
            self.env.render()

        # Select and execute action
        action = self.select_action(self.state, epsilon)
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def select_action(self):
        raise NotImplementedError
