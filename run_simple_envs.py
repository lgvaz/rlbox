import gym
import numpy as np
from utils import SimpleReplayBuffer, get_epsilon_op, egreedy_police
from model import DQN


# Constants
ENV_NAME = 'CartPole-v0'
LEARNING_RATE = 1e-3
NUM_TIMESTEPS = int(1e5)
# NUM_TIMESTEPS = 1
BATCH_SIZE = 32
GAMMA = .9
UPDATE_TARGET_STEPS = 500
FINAL_EPSILON = 0.1
STOP_EXPLORATION = 5000
LOG_STEPS = 1000
MAX_REPLAYS = int(1e4)
MIN_REPLAYS = int(1e3)

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
model = DQN(state_shape, num_actions, LEARNING_RATE)

state = env.reset()
model.target_update()
get_epsilon = get_epsilon_op(FINAL_EPSILON, STOP_EXPLORATION)
# Create logs variables
reward_sum = 0
rewards = []
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
        reward_sum = 0

    # Train
    b_s, b_a, b_r, b_d, b_s_ = buffer.sample(BATCH_SIZE)
    # Compute max-Q using target network
    Q_next = model.target_predict(b_s_)
    Q_next_max = np.max(Q_next, axis=1)
    # Calculate TD target
    td_target = reward + (1 - b_d) * GAMMA * Q_next_max
    # Update weights of main model
    model.fit(b_s, b_a, td_target, verbose=0)

    # Update weights of target model
    if i_step % UPDATE_TARGET_STEPS == 0:
        print('Updating target model...')
        model.target_update()

    # Display logs
    if i_step % LOG_STEPS == 0:
        mean_reward = np.mean(rewards)
        rewards = []
        print('[Step: {}][Mean Reward: {}]'.format(i_step, mean_reward))

