import os
import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from gymmeforce.common.print_utils import Logger


# TODO: Maybe wrap env outside of class??
# TODO: Record frequency as an argument
# TODO: Change video dir when evaluating
class BaseAgent:
    def __init__(self, env_name, log_dir='data/examples',
                 env_wrapper=None, debug=False, **kwargs):
        self.env_name = env_name
        self.log_dir = log_dir
        self.env_wrapper = env_wrapper
        self.logger = Logger(debug)
        self.model = None
        self.sess = None
        self.env_config = {
            'env_name': env_name,
            'env_wrapper': env_wrapper
        }

        env = gym.make(env_name)
        # Adds additional wrappers
        if env_wrapper is not None:
            env = env_wrapper(env)

        # Get env information
        state = env.reset()
        self.env_config['state_shape'] = np.squeeze(state).shape
        env.close()
        # Discrete or continuous actions?
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.env_config['action_space'] = 'discrete'
            self.env_config['num_actions'] = env.action_space.n
        else:
            self.env_config['action_space'] = 'continuous'
            self.env_config['num_actions'] = env.action_space.shape[0]
            self.env_config['action_low_bound'] = env.action_space.low
            self.env_config['action_high_bound'] = env.action_space.high
        # If input is an image defaults to uint8, else defaults to float32
        # TODO: Change this? Only for DQN?
        if len(self.env_config['state_shape']) == 3:
            self.env_config['input_type'] = tf.uint8
        else:
            self.env_config['input_type'] = tf.float32

    def _create_env(self, monitor_dir, record_freq=None, max_episode_steps=None):
        monitor_path = os.path.join(self.log_dir, monitor_dir)
        env = gym.make(self.env_name)
        if max_episode_steps is not None:
            env._max_episode_steps = max_episode_steps
        monitored_env = wrappers.Monitor(
            env=env,
            directory=monitor_path,
            resume=True,
            video_callable=lambda x: record_freq is not None and x % record_freq == True
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

    def select_action(self, state):
        raise NotImplementedError
