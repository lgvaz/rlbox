class EpisodeRunner:
    def __init__(self, env):
        self.env = env
        self.state = env.reset()

    def run_one_step(self, select_action_fn, render=False):
        if render:
            self.env.render()

        # Select and execute action
        action = select_action_fn(self.state)
        next_state, reward, done, info = self.env.step(action)

        trajectory = {
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

        return trajectory
