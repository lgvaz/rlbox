import os
import gym
import fire
import numpy as np
import tensorflow as tf
from model import DQN
from atari_wrapper import wrap_deepmind

# TODO: Register the Q values as the agent plays, then plot them. (Also store the states)

def evaluate(env, sess, model, render=False):
    state = env.reset()
    reward_sum = 0
    while True:
        if render:
            env.render()
        # Choose best action
        if np.random.random() <= 0.05:
            action = env.action_space.sample()
        else:
            Q_values = model.predict(sess, state[np.newaxis])
            action = np.argmax(Q_values)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        state = next_state
        if done:
            state = env.reset()
            return reward_sum

def setup(env_name, log_dir, atari_wrap=True, render=True, record=False):
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
    # env._max_episode_steps = 2000
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Create model
    # TODO: Import model from metagraph
    model = DQN(state_shape, num_actions, learning_rate=0, clip_norm=10)

    # Reload graph
    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None)
    with sv.managed_session() as sess:
        while True:
            reward = evaluate(env, sess, model, render=render)
            print('Life reward: {}'.format(reward))


# Maybe fetch q_values after sv has loaded graph
if __name__ == '__main__':
    fire.Fire(setup)
