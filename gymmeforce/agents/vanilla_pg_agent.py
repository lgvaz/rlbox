import os
import gym
import numpy as np
import tensorflow as tf
import itertools
from gymmeforce.agents import BatchAgent
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import discounted_sum_rewards


class VanillaPGAgent(BatchAgent):
    '''
    Vanilla Policy Gradient

    Args:
    	env_name: Gym environment name

    Keyword args:
        normalize_advantages: Whether or not to normalize advantages (default False)
        use_baseline: Whether or not to subtract a baseline(NN representing the
            value function) from the returns (default True)
        entropy_coef: Entropy penalty added to the loss (default 0.0)
        policy_graph: Function returning a tensorflow graph representing the policy
            (default None)
        value_graph: Function returning a tensorflow graph representing the value function
            (default None)
        log_dir: Directory used for writing logs (default 'logs/examples')
    '''
    def __init__(self, env_name, normalize_advantages, **kwargs):
        super(VanillaPGAgent, self).__init__(env_name, **kwargs)
        self.model = self._create_model(**kwargs)
        self.normalize_advantages = normalize_advantages

    def _create_model(self, **kwargs):
        return VanillaPGModel(self.env_config, **kwargs)

    def _add_discounted_returns(self, trajectory, gamma):
        discounted_returns = discounted_sum_rewards(trajectory['rewards'], gamma)
        trajectory['returns'] = discounted_returns

    def _add_generalized_advantage_estimation(self, trajectory, gamma, gae_lambda):
        assert self.model.use_baseline, 'GAE can only be used with baseline'
        baseline = self.model.compute_baseline(self.sess, trajectory['states'])
        tds = trajectory['rewards'] + gamma * np.append(baseline[1:], 0) - baseline
        trajectory['advantages'] = discounted_sum_rewards(tds, gamma * gae_lambda)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self, learning_rate, max_iters=-1, max_episodes=-1, max_steps=-1, use_gae=True,
              gamma=0.99, gae_lambda=0.96, timesteps_per_batch=2000, num_epochs=1,
              batch_size=64, record_freq=None, max_episode_steps=None):
        self._maybe_create_tf_sess()
        self.logger.add_tf_writer(self.sess, self.model.summary_scalar)
        monitored_env, env = self._create_env(monitor_dir='videos/train',
                                              record_freq=record_freq,
                                              max_episode_steps=max_episode_steps)

        for i_iter in itertools.count():
            # Generate policy rollouts
            trajectories = self.generate_batch(env, timesteps_per_batch)
            for trajectory in trajectories:
                self._add_discounted_returns(trajectory, gamma)
                if use_gae:
                    self._add_generalized_advantage_estimation(trajectory, gamma, gae_lambda)
                else:
                    if self.model.use_baseline:
                        baseline = self.model.compute_baseline(self.sess,
                                                               trajectory['states'])
                        trajectory['advantages'] = trajectory['returns'] - baseline
                    else:
                        trajectory['advantages'] = trajectory['returns']

            states = np.concatenate([trajectory['states'] for trajectory in trajectories])
            actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
            rewards = np.concatenate([trajectory['rewards'] for trajectory in trajectories])
            returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])
            advantages = np.concatenate([trajectory['advantages'] for trajectory in trajectories])

            if self.normalize_advantages:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-7)

            # Update global step
            num_steps = len(rewards)
            self.model.increase_global_step(self.sess, num_steps)

            self.model.fit(self.sess, states, actions, returns, advantages, learning_rate,
                           num_epochs=num_epochs, batch_size=batch_size, logger=self.logger)

            # Logs
            ep_rewards = monitored_env.get_episode_rewards()
            i_episode = len(ep_rewards)
            num_episodes = len(trajectories)
            i_step = self.model.get_global_step(self.sess)
            self.logger.add_log('Reward Mean', np.mean(ep_rewards[-num_episodes:]))
            self.model.write_logs(self.sess, self.logger)
            self.logger.add_log('Learning Rate', learning_rate, precision=4)
            self.logger.timeit(num_steps)
            self.logger.log('Iter {} | Episode {} | Step {}'.format(i_iter, i_episode, i_step))

            # Check for termination
            if (i_iter // max_iters >= 1
                or i_episode // max_episodes >= 1
                or i_step // max_steps >= 1):
                break

        # Save
        self.model.save(self.sess)
