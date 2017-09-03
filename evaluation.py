import os
import gym
import fire
import numpy as np
import tensorflow as tf
from model import DQN
from utils import create_q_values_op, discounted_sum_rewards
from atari_wrapper import wrap_deepmind

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('ggplot')

class Agent():
    def __init__(self, env, sess, q_func, render=True, live_plot=True):
        self.env = env
        self.sess = sess
        self.q_func = q_func
        self.render = render
        self.live_plot = live_plot
        self.action_meanings = env.unwrapped.get_action_meanings()

    def _play_one_step(self):
        if self.render:
            self.env.render()
        # Add some noise so deterministic envs dont roll all the same
        Q_values = self.q_func(self.state[np.newaxis])
        if np.random.random() <= 0.01:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Q_values)

        # Execute action
        next_state, reward, done, _ = self.env.step(action)

        # Update state
        if done:
            self.state = self.env.reset()
        else:
            self.state = next_state

        return Q_values, reward, done

    def _play_one_life(self):
        Q_values, reward, done = self._play_one_step()
        # The value of the state in just max Q
        self.state_values.append(np.max(Q_values))
        self.rewards.append(reward)

        # If done, stop animation
        if done:
            self.anim.event_source.stop()
            plt.close()

        return self.state_values, np.squeeze(Q_values), self.action_meanings

    def play_one_life(self):
        self.state = env.reset()
        self.state_values = []
        self.rewards = []
        # Live plotting setup
        fig, animate = create_animate_func(self._play_one_life)
        self.anim = animation.FuncAnimation(fig, animate, interval=1)
        # Program will stay here until plt.close() is called
        plt.show()
        # Plot over all lifetime
        discounted_rewards = discounted_sum_rewards(self.rewards)
        plt.plot(self.state_values, label='Estimated return')
        plt.plot(discounted_rewards, label='Real return')
        plt.legend()
        plt.show()


def create_animate_func(func, window=100):
    '''
    Create a live matplotlib plot

    Args:
        func: function to animate
        window: maximum points shown in plot
    '''
    fig, ax = plt.subplots(2, 1)

    def animate(i):
        state_values, q_values, action_meanings = func()

        # Redraw
        # TODO: Also plot real return live?
        # Plot value function
        ax[0].clear()
        ax[0].plot(state_values[::-1][:window])
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
        ax[1].set_xticklabels(action_meanings)
        ax[1].xaxis.grid(False)

    return fig, animate



# TODO: Find another way to find if need to atari_wrap
# def setup(env_name, log_dir, num_episodes=10, atari_wrap=True, render=True, record=False):
env_name = 'BreakoutNoFrameskip-v4'
log_dir = 'logs/breakout/v5'
atari_wrap = True
render = True
record = False
live_plot = True

# Create enviroment
env = gym.make(env_name)
# Create videos directory
video_dir = os.path.join(log_dir, 'videos/eval/')
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
env_monitor_wrap = gym.wrappers.Monitor(env, video_dir, resume=True,
                                        video_callable=lambda x: record)
if atari_wrap:
    env = wrap_deepmind(env_monitor_wrap)
else:
    env = env_monitor_wrap

sess = tf.Session()
q_func = create_q_values_op(sess, log_dir)
agent = Agent(env, sess, q_func, render=render)

agent.play_one_life()
