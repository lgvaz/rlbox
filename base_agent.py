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

    # TODO: If this function requires epsilon it can't be on BaseAgent
    def _play_one_step(self, epsilon, render=False):
        if render:
            self.env.render()

        # Select and execute action
        action = self.select_action(self.state, epsilon)
        state_tp1, reward, done, info = self.env.step(action)

        state_t = self.state
        if done:
            self.state = self.env.reset()
        else:
            self.state = state_tp1

        return state_t, state_tp1, action, reward, done, info

    def select_action(self):
        raise NotImplementedError
