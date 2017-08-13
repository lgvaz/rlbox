import gym
import numpy as np
from model import DQN
from utils import egreedy_police

class EnvWatch:
    ''' Create a new instance of the enviroment to run the agent '''
    def __init__(self, env_name, model, render=False):
        self.env = gym.make(env_name)
        self.model = model
        self.render = render

    def run(self):
        ''' Run the agent always selecting the best action '''
        state = self.env.reset()
        reward_sum = 0
        while True:
            if self.render:
                self.env.render()
            # Choose best action
            Q_values = self.model.predict(state[np.newaxis])
            action = egreedy_police(Q_values, epsilon=0)

            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            reward_sum += reward

            # Update state
            state = next_state
            if done:
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
    env.close()

    model = DQN(state_shape, num_actions, LEARNING_RATE, use_huber=USE_HUBER)
    model.load_weights(MODEL_WEIGHTS)

    env_watch = EnvWatch(ENV_NAME, model, render=True)

    while True:
        reward = env_watch.run()
        print('Episode reward: {}'.format(reward))
