import os
import gym
import fire
import numpy as np
import tensorflow as tf
from gymmeforce.common.utils import load_q_func, discounted_sum_rewards
from gymmeforce.wrappers import wrap_deepmind

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('ggplot')


class Agent():
    def __init__(self, env, sess, q_func, render=True, end_life_plot=True,
                 live_plot=True, plot_interval=4):
        self.env = env
        self.sess = sess
        self.q_func = q_func
        self.render = render
        self.end_life_plot = end_life_plot
        self.live_plot = live_plot
        self.plot_interval = plot_interval
        try:
            self.action_meanings = env.unwrapped.get_action_meanings()
        except:
            self.action_meanings = np.arange(env.action_space.n)

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
        for _ in range(self.plot_interval):
            Q_values, reward, done = self._play_one_step()
            # The value of the state in just max Q
            self.state_values.append(np.max(Q_values))
            self.rewards.append(reward)

            # If done, stop animation
            if done:
                if self.live_plot:
                    self.anim.event_source.stop()
                    plt.close()
                break

        return self.state_values, np.squeeze(Q_values), done

    def play_one_life(self):
        self.state = self.env.reset()
        self.state_values = []
        self.rewards = []
        if self.live_plot:
            # Live plotting setup
            fig, animate = create_animate_func(self._play_one_life,
                                               self.action_meanings)
            self.anim = animation.FuncAnimation(fig, animate, interval=1)
            # Program will stay here until plt.close() is called
            plt.show()
        else:
            done = False
            while not done:
                _, _, done = self._play_one_life()
        # Plot over all lifetime
        if self.end_life_plot:
            discounted_rewards = discounted_sum_rewards(self.rewards)
            plt.plot(self.state_values, label='Estimated return')
            plt.plot(discounted_rewards, label='Real return')
            plt.legend()
            plt.show()


def create_animate_func(func, action_meanings, window=100):
    '''
    Create a live matplotlib plot

    Args:
        func: function to animate
        action_meanings: label display when plotting value of actions
        window: maximum points shown in plot
    '''
    fig, ax = plt.subplots(2, 1)

    def animate(i):
        state_values, q_values, _ = func()

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
def setup(env_name, log_dir, num_lives=5, atari_wrap=True,
          end_life_plot=True, live_plot=True, render=True, record=False):

    # Create enviroment
    env = gym.make(env_name)
    # Create videos directory
    video_dir = os.path.join(log_dir, 'videos/eval/')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    env_monitor_wrap = gym.wrappers.Monitor(env, video_dir, resume=True,
                                            video_callable=lambda x: record)
    if atari_wrap:
        env = wrap_deepmind(env_monitor_wrap, frame_stack=4)
    else:
        env = env_monitor_wrap

    with tf.Session() as sess:
        q_func = load_q_func(sess, log_dir)
        agent = Agent(env, sess, q_func, end_life_plot=end_life_plot,
                      live_plot=live_plot, render=render)

        for i_life in range(num_lives):
            print('\rLife {}/{}'.format(i_life + 1, num_lives), end='')
            agent.play_one_life()

    ep_rewards = env_monitor_wrap.get_episode_rewards()
    print('Episodes rewards: {}'.format(ep_rewards))
    print('---------------------')
    print('Rewards mean: {}'.format(np.mean(ep_rewards)))
    print('Maximum reward: {}'.format(np.max(ep_rewards)))
    print('Minimum reward: {}'.format(np.min(ep_rewards)))
    print('---------------------')


if __name__ == '__main__':
    fire.Fire(setup)
