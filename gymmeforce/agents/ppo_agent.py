import os
import gym
import numpy as np
import tensorflow as tf
from gymmeforce.agents import BatchAgent
from gymmeforce.models import PPOModel

class PPOAgent(BatchAgent):
    def __init__(self, env_name, log_dir, policy_graph=None, value_graph=None,
                 input_type=None, env_wrapper=None):
        super(PPOAgent, self).__init__(env_name, log_dir, env_wrapper)

        self.model = PPOModel(self.env_config, policy_graph, value_graph, input_type, log_dir)

    def select_action(self, state):
        return self.model.select_action(self.sess, state)

    def train(self):
        self._maybe_create_tf_sess()
        # env = gym.make(self.env_config['env_name'])
        monitored_env, env = self._create_env(os.path.join(self.log_dir, 'videos/train'))

        for _ in range(20):
            # Generate policy rollouts
            trajectories = self.generate_batch(env, 1000)
            states = np.concatenate([trajectory['states'] for trajectory in trajectories])
            actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
            returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])
            # Train
            # returns = (returns - np.mean(returns) / (np.std(returns) + 1e-7))
            self.model.train(self.sess, states, actions, returns,
                             policy_learning_rate=1e-4, vf_learning_rate=5e-3, num_epochs=10)

            print(np.mean(monitored_env.get_episode_rewards()[-200:]))
