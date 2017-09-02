import os
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

    def add(self, state, action, reward, done):
        if not self.initialized:
            self.initialized = True
            self.states = np.empty([self.maxlen] + list(state.shape), dtype=np.uint8)
            self.actions = np.empty([self.maxlen], dtype=np.int32)
            self.rewards = np.empty([self.maxlen], dtype=np.float32)
            self.dones = np.empty([self.maxlen], dtype=np.bool)

        # Store state
        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.dones[self.current_idx] = done

        self.current_idx = (self.current_idx + 1) % self.maxlen
        self.current_len = min(self.current_len + 1, self.maxlen)

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


def create_q_values_op(sess, log_dir):
    ''' Returns a function that computes the q_values '''
    # Import model from metagraph
    model_path = tf.train.latest_checkpoint(log_dir)
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    # Fetch tensors
    q_values_tensor = tf.get_collection('q_values')[0]
    state_input_ph = tf.get_collection('state_input')[0]

    def compute_q_values(state):
        q_values = sess.run(q_values_tensor,
                            feed_dict={state_input_ph: state})
        return q_values

    return compute_q_values


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
    epsilon_steps = []

    def get_epsilon(step):
        if step <= stop_exploration:
            return epsilon_step * step + epsilon_start
        else:
            return epsilon_final

    return get_epsilon


def piecewise_linear(boundaries, values, initial_value=1):
    ''' Linear interpolates between boundaries '''
    boundaries = [0] + boundaries
    final_epsilons = [initial_value * value for value in values]
    final_epsilons = [initial_value] + final_epsilons

    decay_steps = [end_step - start_step for start_step, end_step
                   in zip(boundaries[:-1], boundaries[1:])]

    decay_rates = [- (start_e - final_e) / decay_step
                   for start_e, final_e, decay_step
                   in zip(final_epsilons[:-1], final_epsilons[1:], decay_steps)]

    def get_epsilon(x):
        for boundary, x0, m, y0 in zip(boundaries[1:], boundaries[:-1], decay_rates, final_epsilons):
            if x <= boundary:
                return m * (x - x0) + y0

        # Outside of boundary
        return final_epsilons[-1]

    return get_epsilon


def egreedy_police(Q_values, epsilon):
    ''' Choose an action based on a egreedy police '''
    if np.random.random() <= epsilon:
        num_actions = len(np.squeeze(Q_values))
        return np.random.choice(np.arange(num_actions))
    else:
        return np.argmax(np.squeeze(Q_values))
