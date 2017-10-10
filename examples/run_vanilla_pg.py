import gym
import numpy as np
import matplotlib.pyplot as plt
from gymmeforce.agents import VanillaPGAgent
# env_name = 'CartPole-v0'
# env_name = 'LunarLander-v2'
# env_name = 'HalfCheetah-v1'
env_name = 'InvertedPendulum-v1'
log_dir = 'tests/tests/2'

env = gym.make(env_name)

agent = VanillaPGAgent(env_name, log_dir, normalize_baseline=False)

agent.train(5e-3, 5e-3, max_iters=100, max_episodes=-1, timesteps_per_batch=2000, num_epochs=1)


trajectories = agent.generate_batch(env, 1)

states = np.concatenate([trajectory['states'] for trajectory in trajectories])
actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])
norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-7)

preds = agent.model.predict_states_value(agent.sess, states)

plt.plot(preds, label='preds')
plt.plot(norm_returns, label='norm_returns')
plt.plot(returns, label='returns')

plt.legend()
plt.show()

agent.model.train(agent.sess, states, actions, returns, 1e-4, 5e-3)

states = np.concatenate([trajectory['states'] for trajectory in trajectories])
actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
returns = np.concatenate([trajectory['returns'] for trajectory in trajectories])
norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-7)

preds = agent.model.predict_states_value(agent.sess, states)

plt.plot(preds, label='preds')
plt.plot(norm_returns, label='norm_returns')
plt.plot(returns, label='returns')

plt.legend()
plt.show()

