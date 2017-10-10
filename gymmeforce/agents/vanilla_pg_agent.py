import os
import gym
import numpy as np
import tensorflow as tf
import itertools
from gymmeforce.agents import BatchAgent
from gymmeforce.models.vanilla_pg_model2 import VanillaPGModel

class VanillaPGAgent(BatchAgent):
    def __init__(self, env_name, log_dir, normalize_baseline=True,
                 policy_graph=None, value_graph=None, input_type=None, env_wrapper=None):
        super(VanillaPGAgent, self).__init__(env_name, log_dir, env_wrapper)

        self.model = VanillaPGModel(self.env_config,
                                    # policy_graph=policy_graph,
                                    # value_graph=value_graph,
                                    log_dir=log_dir)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self, policy_learning_rate, vf_learning_rate, max_iters=-1, max_episodes=-1, max_steps=-1, rew_discount_factor=0.99, timesteps_per_batch=2000, num_epochs=10, batch_size=64):
        self._maybe_create_tf_sess()
        monitored_env, env = self._create_env(os.path.join(self.log_dir, 'videos/train'))

        i_step = 0
        for i_iter in itertools.count():
            # Generate policy rollouts
            trajectories = self.generate_batch(env, timesteps_per_batch, gamma=rew_discount_factor)
            states = np.concatenate([trajectory['states'] for trajectory in trajectories])
            actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
            rewards = np.concatenate([trajectory['rewards'] for trajectory in trajectories])
            returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])

            # Train
            # policy_lr = policy_learning_rate(i_iter)
            # self.model.train(self.sess, states, actions, returns, policy_learning_rate,
                             # vf_learning_rate, num_epochs=num_epochs, batch_size=batch_size, logger=self.logger)
            self.model.fit(self.sess, states, actions, returns)

            # Logs
            ep_rewards = monitored_env.get_episode_rewards()
            i_episode = len(ep_rewards)
            num_episodes = len(trajectories)
            num_steps = len(rewards)
            i_step += num_steps
            self.logger.add_log('Reward Mean [{} episodes]'.format(num_episodes),
                                np.mean(ep_rewards[-num_episodes:]))
            self.logger.timeit(num_steps)
            self.logger.log('Iter {} | Episode {} | Step {}'.format(i_iter, i_episode, i_step))

            # Check for termination
            if (i_iter // max_iters >= 1
                or i_episode // max_episodes >= 1
                or i_step // max_steps >= 1):
                break
