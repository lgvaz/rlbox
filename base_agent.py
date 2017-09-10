import os
import numpy as np
import tensorflow as tf
from gym import wrappers


# TODO: Maybe wrap env outside of class??
# TODO: Record frequency as an argument
class BaseAgent:
    def __init__(self, env, log_dir, env_wrapper=None):
        video_dir = os.path.join(log_dir, 'videos/train')
        self._monitored_env = wrappers.Monitor(env, video_dir,
                                               video_callable=lambda x: x % 500 == 0)
        # Adds additional wrappers
        if env_wrapper is not None:
            self.env = env_wrapper(self._monitored_env)
        else:
            self.env = self._monitored_env

        self.state = self.env.reset()
        self.sess = None

    def _maybe_create_tf_sess(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def select_action(self):
        raise NotImplementedError
