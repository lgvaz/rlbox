import random
import numpy as np
from collections import deque
import tensorflow as tf


class SimpleReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def sample(self, batch_size=32):
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))


class ImgReplayBuffer:
    def __init__(self, maxlen, history_length):
        self.initialized = False
        self.maxlen = maxlen
        self.history_length = history_length
        self.current_idx = 0
        self.current_len = 0

    def add_state(self, state):
        if not self.initialized:
            self.initialized = True
            self.states = np.empty([self.maxlen] + list(state.shape), dtype=np.uint8)
            self.actions = np.empty([self.maxlen], dtype=np.int32)
            self.rewards = np.empty([self.maxlen], dtype=np.float32)
            self.dones = np.empty([self.maxlen], dtype=np.bool)

        # Store state
        idx = self.current_idx
        self.states[idx] = state

        self.current_idx = (self.current_idx + 1) % self.maxlen
        self.current_len = min(self.current_len + 1, self.maxlen)

        return idx

    def add_effect(self, idx, action, reward, done):
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

    def last_state(self):
        end_idx = self.current_idx
        start_idx = max(0, end_idx - self.history_length)

        for idx in range(start_idx, end_idx):
            if self.dones[idx]:
                start_idx = idx + 1

        state = self.states[start_idx:end_idx]

        missing_frames = self.history_length - (end_idx - start_idx)
        if missing_frames > 0:
            zero_frames = np.zeros([missing_frames] + list(state.shape[1:]))
            return np.concatenate((zero_frames, state), axis=0).transpose(1, 2, 0)
        else:
            return state.transpose(1, 2, 0)

    def sample(self, batch_size):
        start_idxs, end_idxs = self._generate_idxs(batch_size)

        # TODO: Only splice self.states once to get state and next_state
        states = np.array([self.states[start_idx:end_idx] for
                             start_idx, end_idx in zip(start_idxs, end_idxs)])
        states_next = np.array([self.states[start_idx + 1: end_idx + 1] for
                                  start_idx, end_idx in zip(start_idxs, end_idxs)])
        # Remember that when spilicing the end_idx is not included
        actions = self.actions[end_idxs - 1]
        rewards = self.rewards[end_idxs - 1]
        dones = self.dones[end_idxs - 1]

        return (states.transpose(0, 2, 3, 1),
                states_next.transpose(0, 2, 3, 1),
                actions,
                rewards,
                dones)

    def _generate_idxs(self, batch_size):
        start_idxs = []
        end_idxs = []
        while len(start_idxs) < batch_size:
            start_idx = np.random.randint(0, self.current_len - self.history_length)
            end_idx = start_idx + self.history_length

            # Check if idx was already picked
            if start_idx in start_idxs:
                continue
            # Only the last frame can have done == True
            # TODO: Check if done checking is correct
            for i_idx in range(start_idx, end_idx - 1):
                if self.dones[i_idx] is True:
                    continue

            start_idxs.append(start_idx)
            end_idxs.append(end_idx)

        return np.array(start_idxs), np.array(end_idxs)

def huber_loss(y_true, y_pred, delta=1.):
    '''
    Hubber loss is less sensitive to outliers
    https://en.wikipedia.org/wiki/Huber_loss
    '''
    error = y_true - y_pred
    condition = tf.abs(error) <= delta
    squared_error = 0.5 * tf.square(error)
    linear_error = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(condition, squared_error, linear_error)


def exponential_epsilon_decay(epsilon_final, stop_exploration):
    ''' Calculate epsilon based on an exponential interpolation '''
    epsilon_step = - np.log(epsilon_final) / stop_exploration

    def get_epsilon(step):
        if step <= stop_exploration:
            return np.exp(-epsilon_step * step)
        else:
            return epsilon_final

    return get_epsilon


def linear_epsilon_decay(epsilon_final, stop_exploration, epsilon_start=1):
    ''' Calculates epsilon based on a linear interpolation '''
    epsilon_step = - (epsilon_start - epsilon_final) / stop_exploration

    def get_epsilon(step):
        if step <= stop_exploration:
            return epsilon_step * step + epsilon_start
        else:
            return epsilon_final

    return get_epsilon


def egreedy_police(Q_values, epsilon):
    ''' Choose an action based on a egreedy police '''
    if np.random.random() <= epsilon:
        num_actions = len(np.squeeze(Q_values))
        return np.random.choice(np.arange(num_actions))
    else:
        return np.argmax(np.squeeze(Q_values))

