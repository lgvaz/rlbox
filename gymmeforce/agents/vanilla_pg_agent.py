import os
import gym
import numpy as np
import tensorflow as tf
import itertools
from gymmeforce.agents import BatchAgent
from gymmeforce.models import VanillaPGModel
from gymmeforce.common.utils import discounted_sum_rewards, explained_variance


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

    def _add_discounted_returns(self, trajectory):
        discounted_returns = discounted_sum_rewards(trajectory['rewards'], self.gamma)
        trajectory['returns'] = discounted_returns

    def _add_advantages_and_vtarget(self, trajectory):
        if self.model.use_baseline:
            # This is the classical way to fir vtarget (directly by the return)
            # TODO: Should a option to bootstrap be added?
            trajectory['baseline_target'] = trajectory['returns']
            trajectory['baseline'] = self.model.compute_baseline(self.sess,
                                                                 trajectory['states'])
            trajectory['advantages'] = trajectory['returns'] - trajectory['baseline']
        else:
            trajectory['advantages'] = trajectory['returns']

    def _normalize_advantages(self, trajectory):
        mean_adv = np.mean(trajectory['advantages'])
        std_adv = np.std(trajectory['advantages'])
        trajectory['advantages'] = (trajectory['advantages'] - mean_adv) / (std_adv + 1e-7)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self, learning_rate, max_iters=-1, max_episodes=-1, max_steps=-1,
              gamma=0.99, timesteps_per_batch=2000, num_epochs=1,
              batch_size=64, record_freq=None, max_episode_steps=None):
        self.gamma = gamma
        self._maybe_create_tf_sess()
        self.logger.add_tf_writer(self.sess, self.model.summary_scalar)
        monitored_env, env = self._create_env(monitor_dir='videos/train',
                                              record_freq=record_freq,
                                              max_episode_steps=max_episode_steps)

        for i_iter in itertools.count():
            # Generate policy rollouts
            trajectories = self.generate_batch(env, timesteps_per_batch)
            for trajectory in trajectories:
                self._add_discounted_returns(trajectory)
                self._add_advantages_and_vtarget(trajectory)
                if self.normalize_advantages:
                    self._normalize_advantages(trajectory)

            states = np.concatenate([traj['states'] for traj in trajectories])
            actions = np.concatenate([traj['actions'] for traj in trajectories])
            rewards = np.concatenate([traj['rewards'] for traj in trajectories])
            returns = np.concatenate([traj['returns'] for traj in trajectories])
            advantages = np.concatenate([traj['advantages'] for traj in trajectories])
            # Change to vtarg
            baseline = np.concatenate([traj['baseline'] for traj in trajectories])
            baseline_targets = np.concatenate([traj['baseline_target']
                                               for traj in trajectories])


            # Update global step
            num_steps = len(rewards)
            self.model.increase_global_step(self.sess, num_steps)

            self.model.fit(self.sess, states, actions, baseline_targets, advantages, learning_rate,
                           num_epochs=num_epochs, batch_size=batch_size, logger=self.logger)

            # Logs
            ep_rewards = monitored_env.get_episode_rewards()
            i_episode = len(ep_rewards)
            num_episodes = len(trajectories)
            i_step = self.model.get_global_step(self.sess)
            ev = explained_variance(y_true=baseline_targets, y_pred=baseline)
            self.logger.add_log('Reward Mean', np.mean(ep_rewards[-num_episodes:]))
            self.logger.add_log('baseline/Explained Variance', ev)
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
