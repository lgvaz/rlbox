import os
import gym
import fire
import numpy as np
import tensorflow as tf
from model import DQN
from utils import egreedy_police


def evaluate(env, sess, model, render=False):
    state = env.reset()
    reward_sum = 0
    while True:
        if render:
            env.render()
        # Choose best action
        Q_values = model.predict(sess, state[np.newaxis])
        action = egreedy_police(Q_values, epsilon=0)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        state = next_state
        if done:
            state = env.reset()
            return reward_sum

def setup(env_name, log_dir, save_videos=False):
    # Create enviroment
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Create videos directory
    render = True
    if save_videos:
        render = False
        video_dir = os.path.join(log_dir, 'videos/eval')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = gym.wrappers.Monitor(env, video_dir,
                                   video_callable=lambda x: x % 2 == 0)

    # Create model
    # TODO: Import model from metagraph
    model = DQN(state_shape, num_actions, learning_rate=0)

    # Reload graph
    sv = tf.train.Supervisor(logdir=log_dir, summary_op=None)
    with sv.managed_session() as sess:
        while True:
            reward = evaluate(env, sess, model, render=False)
            print('Episode reward: {}'.format(reward))

# Maybe fetch q_values after sv has loaded graph
if __name__ == '__main__':
    fire.Fire(setup)
