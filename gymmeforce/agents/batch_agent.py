import numpy as np
from gymmeforce.agents import BaseAgent
from gymmeforce.common.utils import discounted_sum_rewards


class BatchAgent(BaseAgent):
    def __init__(self, env_name, log_dir, env_wrapper=None, debug=False):
        super(BatchAgent, self).__init__(env_name, log_dir, env_wrapper, debug=debug)

    def _run_episode(self, env, render=False):
        state = env.reset()
        done = False
        states, actions, rewards = [], [], []

        while not done:
            if render:
                env.render()
            states.append(state)
            # Select and execute action
            action = self.select_action(state)
            if self.env_config['action_space'] == 'continuous':
                action = action[0]
            state, reward, done, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)

        trajectory = {'states': np.array(states),
                      'actions': np.array(actions),
                      'rewards': np.array(rewards)}

        return trajectory


    def generate_batch(self, env, batch_timesteps, gamma=0.99):
        total_steps = 0
        trajectories = []

        while total_steps < batch_timesteps:
            trajectory = self._run_episode(env)
            # Discounted sum of rewards
            discounted_returns = discounted_sum_rewards(trajectory['rewards'], gamma)
            trajectory['returns'] = discounted_returns

            trajectories.append(trajectory)
            total_steps += trajectory['rewards'].shape[0]

        return trajectories
