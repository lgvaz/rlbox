import numpy as np
import tensorflow as tf
from utils import piecewise_linear
from model import DQNModel

class DQNAgent:
    def __init__(self, env, log_dir, graph=None, input_type=None, double=False):
        self.env = env
        self.state = env.reset()
        self.log_dir = log_dir
        self.sess = None
        state_shape = (env.observation_space.shape)
        num_actions = env.action_space.n

        self.model = DQNModel(state_shape, num_actions, graph, double=double)

    def _maybe_create_tf_sess(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _play_one_step(self, epsilon, render):
        if render:
            self.env.render()

        action = self.select_action(self.state, epsilon)

        # Execute action
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def select_action(self, state, epsilon):
        # Select action based on an egreedy policy
        if np.random.random() <= epsilon:
            action = self.env.action_space.sample()
        else:
            Q_values = self.model.predict(self.sess, self.state[np.newaxis])
            action = np.argmax(Q_values)

        return action

    def play_one_life(self, epsilon=0.01, render=True):
        self._maybe_create_tf_sess()
        done = False
        while not done:
            next_state, reward, done, _ = self._play_one_step(epsilon, render)
            # Update state
            if done:
                self.state = env.reset()
            else:
                self.state = next_state

    #TODO: Define how pass lr_func, get_epsilon
    # def train(self, num_steps, buffer_size, lr_func=piecewise_linear, eps_func=piecewise_linear, lr_schedule=)
