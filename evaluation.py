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
from matplotlib import style
style.use('ggplot')

# TODO: Register the Q values as the agent plays, then plot them. (Also store the states)

def evaluate(env, sess, q_func, render=False):
    state = env.reset()
    reward_sum = 0
    ep_q_values = []
    while True:
        if render:
            env.render()
        # Choose best action
        # Adding random action here just so deterministic envs don't rollout all the same
        if np.random.random() <= 0.01:
            action = env.action_space.sample()
        else:
            Q_values = q_func(state[np.newaxis])
            ep_q_values.append(np.max(Q_values))
            action = np.argmax(Q_values)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        state = next_state
        if done:
            plt.plot(ep_q_values)
            plt.show()
            state = env.reset()
            return reward_sum

def animate_op(env, sess, q_func, render=False):
    state = env.reset()
    ep_q_values = []

    def run_one_step():
        if np.random.random() <= 0.01:
            action = env.action_space.sample()
        else:
            Q_values = q_func(state[np.newaxis])
            ep_q_values.append(np.max(Q_values))
            action = np.argmax(Q_values)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        if done:
            state = env.reset()
        else:
            state = next_state

        return ep_q_values

    return run_one_step

class Agent():
    def __init__(self, env, sess, q_func, render=True, live_plot=True):
        self.env = env
        self.sess = sess
        self.q_func = q_func
        self.render = render
        self.live_plot = live_plot

        self.state = env.reset()
        self.past_q_values = []
        self.rewards = []

        if live_plot:
            fig, animate = create_animate_func(self._play_one_life)
            self.anim = animation.FuncAnimation(fig, animate, interval=1)

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

    def play_one_step(self):
        Q_values, reward, done = self._play_one_step()
        self.past_q_values.append(np.max(Q_values))

        return self.past_q_values

    def _play_one_life(self):
        Q_values, reward, done = self._play_one_step()
        self.past_q_values.append(np.max(Q_values))
        self.rewards.append(reward)

        # If done, stop animation
        if done:
            self.anim.event_source.stop()
            plt.close()

        return self.past_q_values

    def play_one_life(self):
        # Live plotting (until end of life)
        plt.show()
        # Plot over all lifetime
        discounted_rewards = discounted_sum_rewards(self.rewards)
        plt.plot(self.past_q_values, label='Estimated return')
        plt.plot(discounted_rewards, label='Real return')
        plt.show()


def create_animate_func(func, window=100):
    '''
    Create a live matplotlib plot

    Args:
        func: function to animate
        window: maximum points shown in plot
    '''
    fig, ax = plt.subplots()

    def animate(i):
        y = func()[::-1][:window]

        # Redraw
        ax.clear()
        ax.set_title('Value function')
        ax.plot(y)

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
