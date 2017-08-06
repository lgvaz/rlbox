import os
import gym
import numpy as np
from utils import SimpleReplayBuffer, get_epsilon_op, egreedy_police, create_summary
from model import DQN



# Constants
ENV_NAME = 'CartPole-v0'
LEARNING_RATE = 1e-3
USE_HUBER = True
NUM_TIMESTEPS = int(1e7)
BATCH_SIZE = 64
GAMMA = .99
UPDATE_TARGET_STEPS = int(5000)
FINAL_EPSILON = 0.01
STOP_EXPLORATION = int(30000)
LOG_STEPS = int(1000)
MAX_REPLAYS = int(100000)
MIN_REPLAYS = int(10000)
LOG_DIR = 'logs/cart_pole/v7'

# # Constants
# ENV_NAME = 'LunarLander-v2'
# LEARNING_RATE = 3e-4
# USE_HUBER = False
# NUM_TIMESTEPS = int(1e7)
# BATCH_SIZE = 64
# GAMMA = .99
# UPDATE_TARGET_STEPS = int(1e4)
# FINAL_EPSILON = 0.01
# STOP_EXPLORATION = int(1e6)
# LOG_STEPS = int(1e4)
# MAX_REPLAYS = int(1e6)
# MIN_REPLAYS = int(5e4)
# LOG_DIR = 'logs/lunar_lander'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Create new enviroment
env = gym.make(ENV_NAME)

buffer = SimpleReplayBuffer(maxlen=MAX_REPLAYS)
# Populate replay memory
print('Populating replay buffer...')
state = env.reset()
for _ in range(MIN_REPLAYS):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    buffer.add(state, action, reward, done, next_state)

    # Update state
    state = next_state
    if done:
        state = env.reset()

# Create DQN model
state_shape = env.observation_space.shape
num_actions = env.action_space.n
model = DQN(state_shape, num_actions, LEARNING_RATE, use_huber=USE_HUBER)

state = env.reset()
model.target_update()
get_epsilon = get_epsilon_op(FINAL_EPSILON, STOP_EXPLORATION)
# Create logs variables
summary = create_summary(LOG_DIR)
reward_sum = 0
rewards = []
losses = []
for i_step in range(1, NUM_TIMESTEPS + 1):
    # env.render()
    # Choose an action
    Q_values = model.predict(state[np.newaxis])
    epsilon = get_epsilon(i_step)
    action = egreedy_police(Q_values, epsilon)

    # Execute action
    next_state, reward, done, _ = env.step(action)
    reward_sum += reward

    # Store experience
    buffer.add(state, action, reward, done, next_state)

    # Update state
    state = next_state
    if done:
        state = env.reset()
        rewards.append(reward_sum)
        summary('reward', reward_sum, i_step)
        reward_sum = 0

    # Train
    b_s, b_a, b_r, b_d, b_s_ = buffer.sample(BATCH_SIZE)
    # Compute max-Q using target network
    Q_next = model.target_predict(b_s_)
    Q_next_max = np.max(Q_next, axis=1)
    # Calculate TD target
    td_target = reward + (1 - b_d) * GAMMA * Q_next_max
    # Update weights of main model
    loss = model.fit(b_s, b_a, td_target)
    losses.append(loss)

    # Update weights of target model
    if i_step % UPDATE_TARGET_STEPS == 0:
        print('Updating target model...')
        model.target_update()

    # Display logs
    if i_step % LOG_STEPS == 0:
        mean_reward = np.mean(rewards)
        rewards = []
        mean_loss = np.mean(losses)
        losses = []
        summary('mean_reward', mean_reward, i_step)
        summary('loss', mean_loss, i_step)
        print('[Step: {}][Mean Reward: {:.2f}][Epsilon: {:.2f}]'.format(i_step, mean_reward, epsilon))
