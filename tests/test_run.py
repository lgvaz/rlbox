import gym
from gymmeforce.common.utils import *
from gymmeforce.agents import DQNAgent
from gymmeforce.wrappers import wrap_deepmind

env = gym.make('BreakoutNoFrameskip-v4')
# env = wrap_deepmind(env)

agent = DQNAgent(env, 'logs/tests/breakout/v1', 4, double=True, env_wrapper=wrap_deepmind)

num_steps = 1e6
eps = linear_decay(.1, 1e6 * .1)

# agent.train(num_steps, 1e-4, eps, 1e5, 1000)
