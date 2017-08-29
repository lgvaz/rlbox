import os
import gym
import numpy as np
import tensorflow as tf
from utils import *
from model import DQN
from atari_wrapper import wrap_deepmind


# Constants
ENV_NAME = 'BreakoutNoFrameskip-v4'
LEARNING_RATE = 1e-3
USE_HUBER = True
NUM_STEPS = int(40e6)
BATCH_SIZE = 32
GAMMA = .99
UPDATE_TARGET_STEPS = int(1e4)
FINAL_EPSILON = 0.1
STOP_EXPLORATION = int(1e6)
LOG_STEPS = int(1e4)
MAX_REPLAYS = int(1e6)
MIN_REPLAYS = int(5e4)
LOG_DIR = 'logs/breakout/v2'
VIDEO_DIR = LOG_DIR + '/videos/train'
LR_DECAY_RATE = 0.05
LR_DECAY_STEPS = 4e6
HISTORY_LENGTH = 4
LEARNING_FREQ = 4
CLIP_NORM = 10

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
env_monitor_wrapped = gym.wrappers.Monitor(env, VIDEO_DIR,
                                           video_callable=lambda x: x % 500 == 0)
env = wrap_deepmind(env_monitor_wrapped, frame_stack=HISTORY_LENGTH)

buffer = ImgReplayBuffer(MAX_REPLAYS, HISTORY_LENGTH)
# Populate replay memory
print('Populating replay buffer...')
state = env.reset()
for _ in range(MIN_REPLAYS):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    buffer.add(state[..., -1], action, reward, done)
    # Update state
    state = next_state
    if done:
        state = env.reset()

# Create DQN model
state_shape = list(env.observation_space.shape)
num_actions = env.action_space.n
model = DQN(state_shape, num_actions, LEARNING_RATE, CLIP_NORM,
            lr_decay_steps=LR_DECAY_STEPS, lr_decay_rate=LR_DECAY_RATE, gamma=GAMMA)

state = env.reset()
get_epsilon = exponential_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
# get_epsilon = linear_epsilon_decay(FINAL_EPSILON, STOP_EXPLORATION)
# Create logs variables
summary_op = model.create_summaries()
reward_sum = 0
rewards = []

sv = tf.train.Supervisor(logdir=LOG_DIR, summary_op=None)
print('Started training...')
with sv.managed_session() as sess:
    global_step = tf.train.global_step(sess, model.global_step_tensor)
    for i_step in range(global_step, NUM_STEPS + 1):
        model.increase_global_step(sess)
        # Choose an action
        epsilon = get_epsilon(i_step)
        if np.random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            Q_values = model.predict(sess, state[np.newaxis])
            action =  np.argmax(np.squeeze(Q_values))

        # Execute action
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward

        # Store experience
        buffer.add(state[..., -1], action, reward, done)

        # Update state
        state = next_state
        if done:
            state = env.reset()
            rewards.append(reward_sum)
            reward_sum = 0

        # Train
        if i_step % LEARNING_FREQ == 0:
            b_s, b_s_, b_a, b_r, b_d = buffer.sample(BATCH_SIZE)
            model.train(sess, b_s, b_s_, b_a, b_r, b_d)

        # Update weights of target model
        if i_step % UPDATE_TARGET_STEPS == 0:
            # print('Updating target model...')
            model.update_target_net(sess)

        # Display logs
        if i_step % LOG_STEPS == 0:
            ep_rewards = env_monitor_wrapped.get_episode_rewards()
            num_episodes = len(ep_rewards)
            mean_ep_rewards = np.mean(ep_rewards[-50:])
            mean_reward = np.mean(rewards)
            rewards = []
            summary_op(sess, sv, b_s, b_s_, b_a, b_r, b_d)
            model.summary_scalar(sess, sv, 'reward_by_episode(unclipped)', mean_ep_rewards)
            model.summary_scalar(sess, sv, 'reward_by_life(clipped)', mean_reward)
            model.summary_scalar(sess, sv, 'epsilon', epsilon)
            print('[Step: {}]'.format(i_step), end='')
            print('[Episode: {}]'.format(num_episodes), end='')
            print('[Epsilon: {:.2f}]'.format(epsilon))
            print('[Life reward: {:.2f}]'.format(mean_reward), end='')
            print('[Episode reward: {:.2f}]'.format(mean_ep_rewards), end='\n\n')
