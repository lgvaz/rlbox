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

    def generate_trajectories(self,
                              ep_runner,
                              timesteps_per_batch=-1,
                              episodes_per_batch=-1,
                              **kwargs):
        assert timesteps_per_batch > -1 or episodes_per_batch > -1, \
        'You must define how many timesteps or episodes will be in each batch'

        total_steps = 0
        trajectories = []

        while True:
            trajectory = ep_runner.run_one_episode(select_action_fn=self.select_action)
            trajectories.append(trajectory)
            total_steps += trajectory['rewards'].shape[0]

            if self.scaler is not None:
                unscaled = np.concatenate(
                    [traj['unscaled_states'] for traj in trajectories])
                self.update_scaler(unscaled)

            if (total_steps // timesteps_per_batch >= 1
                    or len(trajectories) // episodes_per_batch >= 1):
                break

        # Update global step
        self.model.increase_global_step(self.sess, total_steps)
        return trajectories
