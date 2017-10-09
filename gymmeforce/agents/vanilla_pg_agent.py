import os
import gym
import numpy as np
import tensorflow as tf
import itertools
from gymmeforce.agents import BatchAgent
from gymmeforce.models import VanillaPGModel

class VanillaPGAgent(BatchAgent):
    def __init__(self, env_name, log_dir, normalize_baseline=True,
                 policy_graph=None, value_graph=None, input_type=None, env_wrapper=None):
        super(VanillaPGAgent, self).__init__(env_name, log_dir, env_wrapper)

        self.model = VanillaPGModel(self.env_config,
                                    normalize_baseline=normalize_baseline,
                                    policy_graph=policy_graph,
                                    value_graph=value_graph,
                                    input_type=input_type,
                                    log_dir=log_dir)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self, max_iters=-1, max_episodes=-1, max_steps=-1):
        self._maybe_create_tf_sess()
        monitored_env, env = self._create_env(os.path.join(self.log_dir, 'videos/train'))

        i_step = 0
        for i_iter in itertools.count():
            # Generate policy rollouts
            trajectories = self.generate_batch(env, 1000)
            states = np.concatenate([trajectory['states'] for trajectory in trajectories])
            actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
            rewards = np.concatenate([trajectory['rewards'] for trajectory in trajectories])
            returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])

            # Train
            self.model.train(self.sess, states, actions, returns, policy_learning_rate=1e-4,
                             vf_learning_rate=5e-3, num_epochs=10, logger=self.logger)

            # Logs
            ep_rewards = monitored_env.get_episode_rewards()
            i_episode = len(ep_rewards)
            num_episodes = len(trajectories)
            num_steps = len(rewards)
            i_step += num_steps
            self.logger.add_log('Reward Mean [{} episodes]'.format(num_episodes),
                                np.mean(ep_rewards[-num_episodes:]))
            self.logger.timeit(num_steps)
            self.logger.log('Iter: {} | Episode: {} | Step: {}'.format(i_iter, i_episode, i_step))

            # Check for termination
            if (i_iter // max_iters >= 1
                or i_episode // max_episodes >= 1
                or i_step // max_steps >= 1):
                break
