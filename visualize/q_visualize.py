import numpy as np
from gymmeforce.agents import DQNAgent
from gymmeforce.common.utils import discounted_sum_rewards

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('ggplot')

class DQNVisualize(DQNAgent):
    def __init__(self, env_name, log_dir, history_length=4, graph=None,
                 input_type=None, double=False, dueling=False, env_wrapper=None):
        # TODO: Assert that log_dir already exists
        super(DQNVisualize, self).__init__(env_name, log_dir, history_length,
                                           graph, input_type, double, dueling, env_wrapper)
        # Create enviroment
        self.monitored_env, self.env = self._create_env(env_name)
        self.state = self.env.reset()
        try:
            self.action_meanings = self.env.unwrapped.get_action_meanings()
        except:
            self.action_meanings = np.arange(self.env.action_space.n)

    def select_action(self, state, epsilon=0.01):
        Q_values = self.model.predict(self.sess, state[np.newaxis])
        if np.random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Q_values)

        return action, Q_values

    def _play_one_step(self, epsilon, plot_interval=4, render=True):
        for _ in range(plot_interval):
            if render:
                self.env.render()

            # Concatenates <history_length> states
            self.states_history.append(self.state)
            state_hist = self.states_history.get_data()

            # Select and execute action
            action, Q_values = self.select_action(state_hist, epsilon)
            next_state, reward, done, info = self.env.step(action)

            self.state_values.append(np.max(Q_values))
            self.rewards.append(reward)

            if done:
                self.state = self.env.reset()
                self.states_history.reset()
                self.anim.event_source.stop()
                plt.close()
                break
            else:
                self.state = next_state

        return np.squeeze(Q_values), reward, done

    def visualize(self, num_lives=3, render=True):
        self._maybe_create_tf_sess()
        for _ in range(num_lives):
            self.state_values = []
            self.rewards = []
            fig, animate = self.create_animate_func(render=render)
            self.anim = animation.FuncAnimation(fig, animate, interval=1)
            # Program will stay here until plt.close() is called
            plt.show()
            # Compare estimated state value to real return
            discounted_rewards = discounted_sum_rewards(self.rewards)
            plt.plot(self.state_values, label='Estimated return')
            plt.plot(discounted_rewards, label='Real return')
            plt.legend()
            plt.show()

    def create_animate_func(self, epsilon=0.01, window=100, render=True):
        '''
        Create a live matplotlib plot

        Args:
            func: function to animate
            action_meanings: label display when plotting value of actions
            window: maximum points shown in plot
        '''
        fig, ax = plt.subplots(2, 1)

        def animate(i):
            q_values, _, _ = self._play_one_step(epsilon, render=render)

            # TODO: Also plot real return live?
            # Plot value function
            ax[0].clear()
            ax[0].plot(self.state_values[::-1][:window])
            ax[0].set_title('Value function')
            ax[0].set_xlim(0, window)
            ax[0].set_xticks([])
            ax[0].xaxis.grid(False)
            # Plot value of actions
            ind = np.arange(len(q_values))
            width = 0.4
            ax[1].clear()
            ax[1].bar(ind + width / 2, q_values, width)
            ax[1].set_title('Action Values')
            ax[1].set_xticks(ind + width / 2)
            ax[1].set_xticklabels(self.action_meanings)
            ax[1].xaxis.grid(False)

        return fig, animate

if __name__ == '__main__':
    from gymmeforce.wrappers import wrap_deepmind

    # Specify gym enviroment
    env_name = 'SpaceInvadersNoFrameskip-v4'

    # Create agent
    agent = DQNVisualize(env_name=env_name,
                         log_dir='../examples/logs/space_invaders/dueling_ddqn_v0_2',
                         double=True,
                         dueling=True,
                         env_wrapper=wrap_deepmind)

    agent.visualize()
