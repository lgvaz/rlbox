import os
import gym
import numpy as np
import tensorflow as tf
import itertools
from gymmeforce.agents import BatchAgent
# from gymmeforce.models.vanilla_pg_model2 import VanillaPGModel
from gymmeforce.models import VanillaPGModel
from gymmeforce.models import PPOModel

class VanillaPGAgent(BatchAgent):
    '''
    Vanilla Policy Gradient

    Args:
    	env_name: Gym environment name

    Keyword args:
        normalize_advantages: Whether or not to normalize advantages (default False)
        use_baseline: Whether or not to subtract a baseline(NN representing the
            value function) from the returns (default True)
        normalize_baseline: Whether or not to normalize baseline (baseline values are rescaled
            to have the same mean and variance of the returns) (default False)
        entropy_coef: Entropy penalty added to the loss (default 0.0)
        policy_graph: Function returning a tensorflow graph representing the policy
            (default None)
        value_graph: Function returning a tensorflow graph representing the value function
            (default None)
        log_dir: Directory used for writing logs (default 'logs/examples')
    '''
    def __init__(self, env_name, **kwargs):
        super(VanillaPGAgent, self).__init__(env_name, **kwargs)
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        return VanillaPGModel(self.env_config, **kwargs)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self, learning_rate, max_iters=-1, max_episodes=-1, max_steps=-1,
              rew_discount_factor=0.99, timesteps_per_batch=2000, num_epochs=1,
              batch_size=64, record_freq=None, max_episode_steps=None):
        self._maybe_create_tf_sess()
        self.logger.add_tf_writer(self.sess, self.model.summary_scalar)
        monitored_env, env = self._create_env(monitor_dir='videos/train',
                                              record_freq=record_freq,
                                              max_episode_steps=max_episode_steps)

        for i_iter in itertools.count():
            # Generate policy rollouts
            trajectories = self.generate_batch(env, timesteps_per_batch,
                                               gamma=rew_discount_factor)
            states = np.concatenate([trajectory['states'] for trajectory in trajectories])
            actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
            rewards = np.concatenate([trajectory['rewards'] for trajectory in trajectories])
            returns = np.concatenate([trajectory['returns'] for trajectory in trajectories]) 

            # Update global step
            num_steps = len(rewards)
            self.model.increase_global_step(self.sess, num_steps)

            self.model.fit(self.sess, states, actions, returns, learning_rate,
                           num_epochs=num_epochs, batch_size=batch_size, logger=self.logger)

            # Logs
            ep_rewards = monitored_env.get_episode_rewards()
            i_episode = len(ep_rewards)
            num_episodes = len(trajectories)
            i_step = self.model.get_global_step(self.sess)
            self.logger.add_log('Reward Mean', np.mean(ep_rewards[-num_episodes:]))
            self.logger.add_log('Entropy', self.model.policy.entropy(self.sess, states))
            self.logger.add_log('Learning Rate', learning_rate, precision=5)
            self.logger.timeit(num_steps)
            self.logger.log('Iter {} | Episode {} | Step {}'.format(i_iter, i_episode, i_step))

            # Check for termination
            if (i_iter // max_iters >= 1
                or i_episode // max_episodes >= 1
                or i_step // max_steps >= 1):
                break

        # Save
        self.model.save(self.sess)
