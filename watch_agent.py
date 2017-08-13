import gym
import numpy as np
from model import DQN
from utils import egreedy_police

def watch(env, model):
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        # Choose best action
        Q_values = model.predict(state[np.newaxis])
        action = egreedy_police(Q_values, epsilon=0)

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Update state
        state = next_state
        if done:
            state = env.reset()
            return reward_sum


if __name__ == '__main__':
    # Constants
    ENV_NAME = 'LunarLander-v2'
    MODEL_WEIGHTS = 'logs/lunar_lander/v8/model2_w.h5'
    EPSILON = 0
    USE_HUBER = True
    LEARNING_RATE = 0

    env = gym.make(ENV_NAME)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = DQN(state_shape, num_actions, LEARNING_RATE, use_huber=USE_HUBER)
    model.load_weights(MODEL_WEIGHTS)

    while True:
        reward = watch(env, model)
        print('Episode reward: {}'.format(reward))
