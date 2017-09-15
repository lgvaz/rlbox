import os
import gym
import numpy as np
import tensorflow as tf
from gym import wrappers


# TODO: Maybe wrap env outside of class??
# TODO: Record frequency as an argument
# TODO: Change video dir when evaluating
class BaseAgent:
    def __init__(self, env_name, log_dir, env_wrapper=None):
        self.env_name = env_name
        self.log_dir = log_dir
        self.env_wrapper = env_wrapper

        # Get env information
        env = gym.make(env_name)
        # Adds additional wrappers
        if env_wrapper is not None:
            env = env_wrapper(env)

        state = env.reset()
        self.state_shape = np.squeeze(state).shape
        self.num_actions = env.action_space.n
        env.close()

        self.model = None
        self.sess = None

    def _create_env(self, monitor_dir, record_steps=False):
        monitor_path = os.path.join(self.log_dir, monitor_dir)
        env = gym.make(self.env_name)
        monitored_env = wrappers.Monitor(
            env=env,
            directory=monitor_path,
            resume=True,
            video_callable=lambda x: record_steps and x % record_steps == True
        )
        if self.env_wrapper is not None:
            env = self.env_wrapper(monitored_env)
        else:
            env = monitored_env

        return monitored_env, env


    def _maybe_create_tf_sess(self):
        '''
        Creates a session and loads model from log_dir (if exists)
        '''
        if self.sess is None:
            self.sess = tf.Session()
            self.model.load_or_initialize(self.sess)

    def select_action(self):
        raise NotImplementedError
