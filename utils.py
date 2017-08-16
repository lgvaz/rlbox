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

