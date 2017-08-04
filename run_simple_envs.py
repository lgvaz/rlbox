import gym
from utils import SimpleReplayBuffer


# Constants
ENV_NAME = 'CartPole-v0'

# Create new enviroment
env = gym.make(ENV_NAME)

buffer = SimpleReplayBuffer(maxlen=1000)
# Populate replay memory
state = env.reset()
for _ in range(5000):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    buffer.add(state, action, reward, done, next_state)

    state = next_state

    if done:
        state = env.reset()
