import os
import gym
import numpy as np
import tensorflow as tf
from utils import *
from model import DQN
from evaluation import evaluate


# # Constants
# ENV_NAME = 'CartPole-v0'
# LEARNING_RATE = 1e-3
# USE_HUBER = True
# NUM_STEPS = int(1e5)
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
# NUM_STEPS = int(1e5)
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
LEARNING_RATE = 1e-3
USE_HUBER = True
NUM_STEPS = int(1e6)
BATCH_SIZE = 64
GAMMA = .99
UPDATE_TARGET_STEPS = int(600)
FINAL_EPSILON = 0.1
STOP_EXPLORATION = int(1e5)
LOG_STEPS = int(5e3)
MAX_REPLAYS = int(5e5)
MIN_REPLAYS = int(1e5)
LOG_DIR = 'logs/lunar_lander/v3_0'
VIDEO_DIR = LOG_DIR + '/videos/train'
LR_DECAY_RATE = 0.1


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
    print('Learning rate decay'.format(LR_DECAY_RATE), file=f)

# Create new enviroment
env = gym.make(ENV_NAME)
# Create separete env for running evaluations
# env_eval = gym.make(ENV_NAME)
# env._max_episode_steps = 5000

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
model = DQN(state_shape, num_actions, LEARNING_RATE,
            lr_decay_steps=NUM_STEPS, lr_decay_rate=LR_DECAY_RATE, gamma=GAMMA)

# Record videos
env = gym.wrappers.Monitor(env, VIDEO_DIR,
                           video_callable=lambda count: count % 100 == 0)
state = env.reset()
# get_epsilon = exponential_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
get_epsilon = linear_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
# Create logs variables
summary_op = model.create_summaries()
reward_sum = 0
rewards = []

sv = tf.train.Supervisor(logdir=LOG_DIR, summary_op=None)
print('Started training...')
with sv.managed_session() as sess:
    global_step = tf.train.global_step(sess, model.global_step_tensor)
    for i_step in range(global_step, NUM_STEPS + 1):
        # Choose an action
        Q_values = model.predict(sess, state[np.newaxis])
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
        model.train(sess, b_s, b_s_, b_a, b_r, b_d)

        # Update weights of target model
        if i_step % UPDATE_TARGET_STEPS == 0:
            print('Updating target model...')
            model.update_target_net(sess)

        # Display logs
        if i_step % LOG_STEPS == 0:
            # eval_reward = np.mean([evaluate(env_eval, sess, model) for _ in range(5)])
            mean_reward = np.mean(rewards)
            rewards = []
            summary_op(sess, sv, b_s, b_s_, b_a, b_r, b_d)
            model.summary_scalar(sess, sv, 'reward_train', mean_reward)
            # model.summary_scalar(sess, sv, 'reward_eval', eval_reward)
            model.summary_scalar(sess, sv, 'epsilon', epsilon)
            print('[Step: {}]'.format(i_step), end='')
            print('[Train reward: {:.2f}]'.format(mean_reward), end='')
            # print('[Eval reward: {:.2f}]'.format(eval_reward), end='')
            print('[Epsilon: {:.2f}]'.format(epsilon))
