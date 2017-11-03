import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from scipy.signal import lfilter


class RingBuffer:
    '''
    Similar function of a deque, but returns a numpy array directly
    Used for building an array with <maxlen> sequential states

    Args:
        state_shape: Shape of state (tuple)
        maxlen: How many states to stack
    '''
    def __init__(self, state_shape, maxlen):
        self.state_shape = state_shape
        self.maxlen = maxlen
        self.current_idx = 0
        self.reset()

    def reset(self):
        self.data = np.zeros(((self.maxlen,) + self.state_shape))

    def append(self, data):
        self.data = np.roll(self.data, -1, axis=0)
        self.data[self.maxlen - 1] = np.squeeze(data)

    def get_data(self):
        return self.data.swapaxes(0, -1)


class ReplayBuffer:
    '''
    Memory efficient implementation of replay buffer, storing each state only once.
    Example: Typical use for atari, with each frame being a 84x84 grayscale
             image (uint8), storing 1M frames should use about 7GiB of RAM
             (8 * 64 * 64 * 1M bits)

    Args:
        maxlen: Maximum number of transitions stored
        history_length: Number of sequential states stacked when sampling
        batch_size: Mini-batch size created by sample
    '''
    def __init__(self, maxlen, history_length=1, batch_size=32, n_step=1):
        self.maxlen = maxlen
        self.history_length = history_length
        self.batch_size = batch_size
        self.n_step = n_step
        self.initialized = False
        self.current_idx = 0
        self.current_len = 0

    def add(self, state, action, reward, done):
        if not self.initialized:
            self.initialized = True
            state_shape = np.squeeze(state).shape
            # Allocate memory
            self.states = np.empty((self.maxlen,) + state_shape,
                                   dtype=state.dtype)
            self.actions = np.empty(self.maxlen, dtype=np.int32)
            self.rewards = np.empty(self.maxlen, dtype=np.float32)
            self.dones = np.empty(self.maxlen, dtype=np.bool)

            # Function for selecting multiple slices
            self.states_stride_history = strided_axis0(self.states, self.history_length)
            self.rewards_stride_nstep = strided_axis0(self.rewards, self.n_step)
            self.dones_stride_nstep = strided_axis0(self.dones, self.n_step)

        # Store transition
        self.states[self.current_idx] = np.squeeze(state)
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.dones[self.current_idx] = done

        # Update current position
        self.current_idx = (self.current_idx + 1) % self.maxlen
        self.current_len = min(self.current_len + 1, self.maxlen)

    def sample(self, n_step=None):
        if n_step is None:
            n_step = self.n_step
        start_idxs, end_idxs = self._generate_idxs()
        # Get states
        b_states_t = self.states_stride_history[start_idxs]
        b_states_tp1 = self.states_stride_history[start_idxs + n_step]
        rewards = self.rewards_stride_nstep[end_idxs - 1]
        dones = self.dones_stride_nstep[end_idxs - 1]
        # Remember that when slicing the end_idx is not included
        actions = self.actions[end_idxs - 1]

        return (b_states_t.swapaxes(1, -1),
                b_states_tp1.swapaxes(1, -1),
                actions, rewards[:, :n_step], dones[:, :n_step])

    def _generate_idxs(self):
        start_idxs = np.random.randint(self.current_len
                                       - self.history_length
                                       - self.n_step,
                                       size=self.batch_size)
        end_idxs = start_idxs + self.history_length
        # start_idxs = []
        # end_idxs = []
        # while len(start_idxs) < self.batch_size:
        #     start_idx = np.random.randint(0, self.current_len
        #                                   - self.history_length
        #                                   - self.n_step)
        #     end_idx = start_idx + self.history_length

        #     # Check if idx was already picked
        #     if start_idx in start_idxs:
        #         continue

        #     # Check if state contains frames only from a single episode
        #     if np.any(self.dones[start_idx : end_idx - 1]):
        #         continue

        #     # Valid idx!!
        #     start_idxs.append(start_idx)
        #     end_idxs.append(end_idx)

        return np.array(start_idxs), np.array(end_idxs)


class Scaler(object):
    """
    From: https://github.com/pat-coady/trpo/blob/master/src/utils.py#L13
    Generate scale and offset based on running mean and stddev along axis=0
    offset = running mean
    scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


def strided_axis0(a, L):
    '''
    https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
    '''
    # Store the shape and strides info
    shp = a.shape
    s  = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0]-L+1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0,L)+shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)


def load_q_func(sess, log_dir):
    ''' Returns a function that computes the q_values '''
    # Import model from metagraph
    model_path = tf.train.latest_checkpoint(log_dir)
    print('Loading model from: {}'.format(model_path))
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

    # Fetch tensors
    q_values_tensor = tf.get_collection('q_online_t')[0]
    state_input_ph = tf.get_collection('state_input')[0]

    def compute_q_values(state):
        q_values = sess.run(q_values_tensor,
                            feed_dict={state_input_ph: state})
        return q_values

    return compute_q_values


def exponential_decay(epsilon_final, stop_exploration):
    ''' Calculate epsilon based on an exponential interpolation '''
    epsilon_step = - np.log(epsilon_final) / stop_exploration

    def get_epsilon(step):
        if step <= stop_exploration:
            return np.exp(-epsilon_step * step)
        else:
            return epsilon_final

    return get_epsilon


def linear_decay(epsilon_final, stop_exploration, epsilon_start=1):
    ''' Calculates epsilon based on a linear interpolation '''
    epsilon_step = - (epsilon_start - epsilon_final) / stop_exploration
    epsilon_steps = []

    def get_epsilon(step):
        if step <= stop_exploration:
            return epsilon_step * step + epsilon_start
        else:
            return epsilon_final

    return get_epsilon


def piecewise_linear_decay(boundaries, values, initial_value=1):
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


def discounted_sum_rewards(rewards, gamma=0.99):
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]


def calculate_n_step_return(rewards, dones, gamma=0.99):
    done_idx = np.where(dones == 1)[0]
    done = False
    if done_idx:
        rewards = rewards[:done_idx[0] + 1]
        done = True

    return discounted_sum_rewards(rewards, gamma)[0], done


def tf_copy_params_op(from_scope, to_scope, soft_update=1.):
    # Get variables within defined scope
    from_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
    to_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)
    # Create operations that copy the variables
    op_holder = [to_scope_var.assign(soft_update * from_scope_var
                                     + (1 - soft_update) * to_scope_var)
                 for from_scope_var, to_scope_var in zip(from_scope_vars, to_scope_vars)]

    return op_holder


def explained_variance(y_true, y_pred):
    return 1 - np.var(y_true - y_pred) / np.var(y_true)
