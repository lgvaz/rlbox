import os
import gym
import numpy as np
from utils import *
from model import DQN
from watch_agent import EnvWatch


# # Constants
# ENV_NAME = 'CartPole-v0'
# LEARNING_RATE = 1e-3
# USE_HUBER = True
# NUM_TIMESTEPS = int(1e5)
# BATCH_SIZE = 64
# GAMMA = .99
# UPDATE_TARGET_STEPS = int(500)
# FINAL_EPSILON = 0.02
# STOP_EXPLORATION = int(1e4)
# LOG_STEPS = int(2000)
# MAX_REPLAYS = int(5e4)
# MIN_REPLAYS = int(1e4)
# LOG_DIR = 'logs/cart_pole/v28'
# VIDEO_DIR = LOG_DIR + '/videos'


# # Constants
# ENV_NAME = 'MountainCar-v0'
# LEARNING_RATE = 1e-3
# USE_HUBER = True
# NUM_TIMESTEPS = int(1e5)
# BATCH_SIZE = 64
# GAMMA = .99
# UPDATE_TARGET_STEPS = int(500)
# FINAL_EPSILON = 0.1
# STOP_EXPLORATION = int(1e4)
# LOG_STEPS = int(2000)
# MAX_REPLAYS = int(5e4)
# MIN_REPLAYS = int(1e4)
# LOG_DIR = 'logs/mountain_car/v1'
# VIDEO_DIR = LOG_DIR + '/videos'


# Constants
ENV_NAME = 'LunarLander-v2'
LEARNING_RATE = 0.00025
USE_HUBER = True
NUM_TIMESTEPS = int(1e6)
BATCH_SIZE = 64
GAMMA = .99
UPDATE_TARGET_STEPS = int(700)
FINAL_EPSILON = 0.05
STOP_EXPLORATION = int(1e5)
LOG_STEPS = int(5e3)
MAX_REPLAYS = int(5e5)
MIN_REPLAYS = int(1e5)
LOG_DIR = 'logs/lunar_lander/v9'
VIDEO_DIR = LOG_DIR + '/videos'


# Create log directory
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

with open(LOG_DIR + '/parameters.txt', 'w') as f:
    print('Learning rate: {}'.format(LEARNING_RATE), file=f)
    print('Loss function: {}'.format(['MSE', 'Huber'][USE_HUBER]), file=f)
    print('Target update steps: {}'.format(UPDATE_TARGET_STEPS), file=f)
    print('Final epsilon: {}'.format(FINAL_EPSILON), file=f)
    print('Stop exploration: {}'.format(STOP_EXPLORATION), file=f)
    print('Memory size: {}'.format(MAX_REPLAYS), file=f)

# Create new enviroment
env = gym.make(ENV_NAME)
# env._max_episode_steps = 5000
# Create DQN model
state_shape = env.observation_space.shape
num_actions = env.action_space.n
model = DQN(state_shape, num_actions, LEARNING_RATE, use_huber=USE_HUBER)
# Create env for evaluation
env_watch = EnvWatch(ENV_NAME, model)

# Populate replay memory
print('Populating replay buffer...')
buffer = SimpleReplayBuffer(maxlen=MAX_REPLAYS)
state = env.reset()
for _ in range(MIN_REPLAYS):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    buffer.add(state, action, reward, done, next_state)

    # Update state
    state = next_state
    if done:
        state = env.reset()

# Record videos
env = gym.wrappers.Monitor(env, VIDEO_DIR,
                           video_callable=lambda count: count % 300 == 0)

state = env.reset()
model.target_update()
# get_epsilon = exponential_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
get_epsilon = linear_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
# Create logs variables
summary = create_summary(LOG_DIR)
reward_sum = 0
rewards = []
losses = []

print('Started training...')
for i_step in range(1, NUM_TIMESTEPS + 1):
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
    # Q_next = model.predict(b_s_)
    Q_next_max = np.max(Q_next, axis=1)
    # Calculate TD target
    td_target = b_r + (1 - b_d) * GAMMA * Q_next_max
    # Update weights of main model
    loss = model.fit(b_s, b_a, td_target)
    losses.append(loss)

    # Update weights of target model
    if i_step % UPDATE_TARGET_STEPS == 0:
        print('Updating target model...')
        model.target_update()

    # Display logs
    if i_step % LOG_STEPS == 0:
        # Run evaluation
        ev_reward = env_watch.run()
        mean_reward = np.mean(rewards)
        rewards = []
        mean_loss = np.mean(losses)
        losses = []
        summary('reward_train', mean_reward, i_step)
        summary('reward_evaluation', ev_reward, i_step)
        summary('epsilon', epsilon, i_step)
        summary('loss', mean_loss, i_step)
        summary('Q_max', np.max(Q_next_max), i_step)
        summary('Q_mean', np.mean(Q_next), i_step)
        print('[Step: {}]'.format(i_step), end='')
        print('[Eval Reward: {:.2f}]'.format(ev_reward), end='')
        print('[Mean Reward: {:.2f}]'.format(mean_reward), end='')
        print('[Epsilon: {:.2f}]'.format(epsilon))

model.model.save_weights(LOG_DIR + '/model_w.h5')
