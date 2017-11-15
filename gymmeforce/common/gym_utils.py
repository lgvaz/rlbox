import numpy as np


class EpisodeRunner:
    def __init__(self, env, monitored_env=None, scaler=None):
        self.env = env
        self.monitored_env = monitored_env
        self.scaler = scaler
        self.state = env.reset()

    def run_one_step(self, select_action_fn, render=False):
        if render:
            self.env.render()

        unscaled_state = self.state
        if self.scaler is not None:
            self.state = self.scaler.scale_state(self.state)
        # Select and execute action
        action = select_action_fn(self.state)
        next_state, reward, done, _ = self.env.step(action)

        transition = {
            'unscaled_state': unscaled_state,
            'state': self.state,
            'next_state': next_state,
            'action': action,
            'reward': reward,
            'done': done
        }

        if done:
            self.state = self.env.reset()
        else:
            self.state = next_state

        return transition

    def run_one_episode(self, **kwargs):
        done = False
        transitions = []

        while not done:
            transition = self.run_one_step(**kwargs)
            transitions.append(transition)
            done = transition['done']

        trajectory = {
            key + 's': np.array([t[key] for t in transitions])
            for key in transitions[0]
        }

        return trajectory

    def get_number_steps(self):
        return self.monitored_env.get_total_steps()

    def get_number_episodes(self):
        return self.monitored_env.episode_id

    def get_episode_rewards(self):
        return self.monitored_env.get_episode_rewards()
