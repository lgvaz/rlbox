import gym
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


# TODO: Run evaluation without the need to import model
# Maybe fetch q_values after sv has loaded graph
if __name__ == '__main__':
    # Constants
    ENV_NAME = 'LunarLander-v2'
    LOG_DIR = 'logs/lunar_lander/tensorflow/v0_1'
    EPSILON = 0
    USE_HUBER = True
    LEARNING_RATE = 0

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = DQN(state_shape, num_actions, LEARNING_RATE)

    sv = tf.train.Supervisor(logdir=LOG_DIR, summary_op=None)
    with sv.managed_session() as sess:
        while True:
            reward = evaluate(env, sess, model, render=True)
            print('Episode reward: {}'.format(reward))
