import os

import gym
import numpy as np
import tensorflow as tf
from gym import wrappers

from gymmeforce.common.runner import EpisodeRunner
from gymmeforce.common.print_utils import Logger
from gymmeforce.common.utils import Scaler


# TODO: Maybe wrap env outside of class??
class BaseAgent:
    def __init__(self,
                 env_name,
                 log_dir='data/examples',
                 env_wrapper=None,
                 scale_states=False,
                 debug=False,
                 **kwargs):
        self.env_name = env_name
        self.log_dir = log_dir
        self.env_wrapper = env_wrapper
        self.logger = Logger(debug)
        self.model = None
        self.sess = None
        self.env_config = {'env_name': env_name, 'env_wrapper': env_wrapper}
        self.play_ep_runner = None
        self.train_ep_runner = None

        env = gym.make(env_name)
        # Adds additional wrappers
        if env_wrapper is not None:
            env = env_wrapper.wrap_env(env)

        # Get env information
        state = env.reset()
        self.env_config['state_shape'] = np.squeeze(state).shape
        if state.dtype == np.uint8:
            self.env_config['input_type'] = tf.uint8
        else:
            self.env_config['input_type'] = tf.float32
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

        # TODO: where scaler should be? In base or batch agent?
        self.scaler = Scaler(self.env_config['state_shape']) if scale_states else None

    def _create_env(self, monitor_dir, record_freq=None, max_episode_steps=None,
                    **kwargs):
        monitor_path = os.path.join(self.log_dir, monitor_dir)
        env = gym.make(self.env_name)
        if max_episode_steps is not None:
            env._max_episode_steps = max_episode_steps
        monitored_env = wrappers.Monitor(
            env=env,
            directory=monitor_path,
            resume=True,
            video_callable=lambda x: record_freq is not None and x % record_freq == True)
        if self.env_wrapper is not None:
            env = self.env_wrapper.wrap_env(monitored_env)
        else:
            env = monitored_env

        return monitored_env, env

    def _calculate_learning_rate(self):
        if callable(self.learning_rate):
            lr = self.learning_rate(self.i_step)
        else:
            lr = self.learning_rate

        return lr

    def _maybe_create_tf_sess(self):
        '''
        Creates a session and loads model from log_dir (if exists)
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if self.sess is None:
            self.sess = tf.Session(config=config)
            self.model.load_or_initialize(self.sess)

    def _step_and_check_termination(self):
        self.i_iter += 1
        self.i_episode = self.train_ep_runner.get_number_episodes()
        self.i_step = self.train_ep_runner.get_number_steps()

        # Check for termination
        if (self.i_iter // self.max_iters >= 1 or self.i_episode // self.max_episodes >= 1
                or self.i_step // self.max_steps >= 1):
            return True

        return False

    def select_action(self, state):
        raise NotImplementedError

    def play(self, render=True, record_freq=1, **kwargs):
        self._maybe_create_tf_sess()

        # Create environment
        if self.play_ep_runner is None:
            monitored_env, env = self._create_env(
                monitor_dir='videos/play', record_freq=record_freq, **kwargs)
            self.play_ep_runner = EpisodeRunner(env, monitored_env, self.scaler)

        self.play_ep_runner.run_one_episode(
            render=render, select_action_fn=self.select_action)

    def train(self, max_iters=-1, max_episodes=-1, max_steps=-1, **kwargs):
        # Create Session
        self._maybe_create_tf_sess()
        self.logger.add_tf_writer(self.sess, self.model.summary_scalar)

        # Create environment
        if self.train_ep_runner is None:
            monitored_env, env = self._create_env(monitor_dir='videos/train', **kwargs)
            self.train_ep_runner = EpisodeRunner(env, monitored_env, self.scaler)

        self.max_iters = max_iters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.i_iter = 0
        self.i_episode = 0
        self.i_step = self.model.get_global_step(self.sess)

    def write_logs(self, batch):
        ep_rewards = self.train_ep_runner.monitored_env.get_episode_rewards()

        self.logger.add_log('Reward/Episode (Last 50)', np.mean(ep_rewards[-50:]))
        self.model.write_logs(self.sess, self.logger)
        self.logger.add_log('Learning Rate', self._calculate_learning_rate(), precision=5)
        self.logger.timeit(self.i_step, max_steps=self.max_steps)

    def update_scaler(self, states):
        self.scaler.update(states)

    def save(self):
        self.model.save(self.sess)
