import numpy as np

from gymmeforce.agents import BaseAgent


class BatchAgent(BaseAgent):
    def __init__(self, env_name, **kwargs):
        super(BatchAgent, self).__init__(env_name, **kwargs)

    def _run_episode(self, env, render=False):
        state = env.reset()
        done = False
        states, actions, rewards, unscaled_states = [], [], [], []

        while not done:
            if render:
                env.render()
            unscaled_states.append(state)
            if self.scale_states:
                state = self.scale_state(state)
            states.append(state)
            # Select and execute action
            action = self.select_action(state)
            if self.env_config['action_space'] == 'continuous':
                action = action[0]
            state, reward, done, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)

        trajectory = {
            'states': np.array(states),
            'unscaled_states': np.array(unscaled_states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }

        return trajectory

    def generate_batch(self, env, batch_timesteps):
        total_steps = 0
        trajectories = []

        while total_steps < batch_timesteps:
            trajectory = self._run_episode(env)
            trajectories.append(trajectory)
            total_steps += trajectory['rewards'].shape[0]

            unscaled = np.concatenate([traj['unscaled_states'] for traj in trajectories])
            if self.scale_states:
                self.update_scaler(unscaled)

        return trajectories
