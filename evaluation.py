import os
import gym
import fire
import numpy as np
import tensorflow as tf
from model import DQN
from utils import create_q_values_op
from atari_wrapper import wrap_deepmind

# TODO: Register the Q values as the agent plays, then plot them. (Also store the states)

def evaluate(env, sess, q_func, render=False):
    state = env.reset()
    reward_sum = 0
    while True:
        if render:
            env.render()
        # Choose best action
        # Adding random action here just so deterministic envs don't rollout all the same
        if np.random.random() <= 0.01:
            action = env.action_space.sample()
        else:
            Q_values = q_func(state[np.newaxis])
            action = np.argmax(Q_values)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        state = next_state
        if done:
            state = env.reset()
            return reward_sum

# TODO: Find another way to find if need to atari_wrap
def setup(env_name, log_dir, num_episodes=10, atari_wrap=True, render=True, record=False):
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
    # env._max_episode_steps = 2000
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n


    with tf.Session() as sess:
        get_q_values = create_q_values_op(sess, log_dir)
        while len(env_monitor_wrap.get_episode_rewards()) < num_episodes:
            reward = evaluate(env, sess, get_q_values, render=render)
            print('Life reward(clipped): {}'.format(reward))
        ep_rewards = env_monitor_wrap.get_episode_rewards()
        print('Episodes rewards: {}'.format(ep_rewards))
        print('---------------------')
        print('Rewards mean: {}'.format(np.mean(ep_rewards)))
        print('Maximum reward: {}'.format(np.max(ep_rewards)))
        print('Minimum reward: {}'.format(np.min(ep_rewards)))
        print('---------------------')


# Maybe fetch q_values after sv has loaded graph
if __name__ == '__main__':
    fire.Fire(setup)
