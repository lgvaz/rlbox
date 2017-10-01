import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from gymmeforce.common.utils import discounted_sum_rewards_final_sum
from gymmeforce.common.utils import ReplayBuffer, RingBuffer
from gymmeforce.models import DQNModel
from gymmeforce.agents.base_agent import BaseAgent
from gymmeforce.common.print_utils import print_table


class DQNAgent(BaseAgent):
    def __init__(self, env_name, log_dir, history_length=4, graph=None,
                 input_type=None, double=False, dueling=False, env_wrapper=None):
        super(DQNAgent, self).__init__(env_name, log_dir, env_wrapper)

        # Create underlying model
        self.model = DQNModel(self.state_shape + (history_length,),
                              self.num_actions, graph=graph, double=double,
                              dueling=dueling, log_dir=log_dir)
        self.history_length = history_length
        self.replay_buffer = None

        # Keep track of past states
        self.states_history = RingBuffer(self.state_shape, history_length)

    def _play_one_step(self, env, state, epsilon, render=False):
        if render:
            env.render()

        # Concatenates <history_length> states
        self.states_history.append(state)
        state_hist = self.states_history.get_data()

        # Select and execute action
        action = self.select_action(env, state_hist, epsilon)
        next_state, reward, done, info = env.step(action)

        if done:
            self.states_history.reset()

        return next_state, action, reward, done, info

    def select_action(self, env, state, epsilon):
        # Select action based on an egreedy policy
        if np.random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            Q_values = self.model.predict(self.sess, state[np.newaxis])
            action = np.argmax(Q_values)

        return action

    def play_n_lives(self, num_lives, epsilon=0.01, render=True, record=False):
        monitored_env, env = self._create_env('videos/eval', record)
        state = env.reset()
        self._maybe_create_tf_sess()
        for _ in range(num_lives):
            done = False
            while not done:
                next_state, action, reward, done, _ = self._play_one_step(env, state,
                                                                          epsilon, render)
                # Update state
                if done:
                    state = env.reset()
                else:
                    state = next_state

        # Print logs
        rewards = monitored_env.get_episode_rewards()
        header = '{} Episodes'.format(len(rewards))
        tags = ['Reward Mean (unclipped)',
                'Reward std_dev',
                'Max reward',
                'Min reward']
        values = ['{:.2f}'.format(np.mean(rewards)),
                  '{:.2f}'.format(np.std(rewards)),
                  '{:.2f}'.format(np.max(rewards)),
                  '{:.2f}'.format(np.min(rewards))]
        print_table(tags, values, header=header)


    def train(self, num_steps, n_step, learning_rate, exploration_schedule,
              replay_buffer_size, target_update_freq, target_soft_update=1.,
              gamma=0.99, clip_norm=10, learning_freq=4,
              init_buffer_size=0.05, batch_size=32, log_steps=2e4):
        '''
        Trains the agent following these steps:
            0. Populate replay buffer (init_buffer_size) with transitions of a random agent
            1. Use the current state to calculate Q-values
               and choose an action based on an epsilon-greedy policy
            2. Store experience on the replay buffer
            3. Every <learning_freq> steps sample the buffer
               and performs gradient descent

        Args:
            num_steps: Number of steps to train the agent
            n_step: Number of steps to use reward before bootstraping
            learning_rate: Float or a function that returns a float
                           when called with the current time step as input
                           (see gymmeforce.utils.linear_decay as an example)
            exploration_schedule: Function that returns a float when
                                  called with the current time step as input
                                  (see utils.linear_decay as an example)
            replay_buffer_size: Maximum number of transitions stored on replay buffer
            target_update_freq: Number of steps between each target update
            target_soft_update: Percentage of online weigth value to copy to target on
                                each update, (e.g. 1 makes target weights = online weights)
            gamma: Discount factor on sum of rewards
            clip_norm: Value to clip the gradient so that its L2-norm is less than or
                       equal to clip_norm
            learning_freq: Number of steps between each gradient descent update
            init_buffer_size: Percentage of buffer filled with random transitions
                              before the training starts
            batch_size: Number of samples to use when creating mini-batch from replay buffer
            log_steps: Number of steps between each log status
        '''
        # Create enviroment
        monitored_env, env = self._create_env('videos/train', 500)
        state = env.reset()

        # Create training ops
        self.model.create_training_ops(gamma, clip_norm, n_step, target_soft_update)
        # Create Session
        self._maybe_create_tf_sess()

        # Create replay buffer
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(int(replay_buffer_size),
                                              history_length=self.history_length,
                                              batch_size=batch_size,
                                              n_step=n_step)
            # Populate replay buffer with random agent
            num_init_replays = replay_buffer_size * init_buffer_size
            for i in range(int(num_init_replays)):
                next_state, action, reward, done, _ = self._play_one_step(env, state, epsilon=1.)
                self.replay_buffer.add(state, action, reward, done)

                if done:
                    state = env.reset()
                else:
                    state = next_state

                # Logs
                if i % 100 == 0:
                    print('\rPopulating replay buffer: {:.1f}%'.format(
                        i * 100 / num_init_replays), end='', flush=True)

        print('\rPopulating replay buffer: DONE!')
        print('Started training')
        num_episodes = 0
        reward_sum = 0
        rewards = []
        start_time = time.time()
        # TODO: soft updating here, need to hard copy weights
        self.model.update_target_net(self.sess)
        for i_step in range(1, int(num_steps) + 1):
            epsilon = exploration_schedule(i_step)
            next_state, action, reward, done, _ = self._play_one_step(env, state, epsilon)
            reward_sum += reward

            # Store experience
            self.replay_buffer.add(state, action, reward, done)

            # Update state
            if done:
                state = env.reset()
                rewards.append(reward_sum)
                reward_sum = 0
            else:
                state = next_state

            # Perform gradient descent
            if i_step % learning_freq == 0:
                # Sample replay buffer
                b_s, b_s_, b_a, b_r, b_d = self.replay_buffer.sample()
                # Calculate n_step rewards
                b_r = [discounted_sum_rewards_final_sum(r) for r in b_r]
                # Calculate learning rate
                if callable(learning_rate):
                    lr = learning_rate(i_step)
                else:
                    lr = learning_rate
                self.model.train(self.sess, lr, b_s, b_s_, b_a, b_r, b_d)

            # Update target network
            if i_step % target_update_freq == 0:
                self.model.update_target_net(self.sess)

            if i_step % log_steps == 0:
                # Save model
                self.model.save(self.sess, i_step)
                # Calculate time
                end_time = time.time()
                time_window = end_time - start_time
                steps_sec = log_steps / time_window
                eta_seconds = (num_steps - i_step) / steps_sec
                # Format days, hours, minutes, seconds and remove miliseconds
                eta_formatted = str(timedelta(seconds=eta_seconds)).split('.')[0]
                start_time = end_time
                # Calculate rewards statistics
                ep_rewards = monitored_env.get_episode_rewards()
                num_episodes_old = num_episodes
                num_episodes = len(ep_rewards)
                num_new_episodes = num_episodes - num_episodes_old
                mean_ep_rewards = np.mean(ep_rewards[-num_new_episodes:])
                mean_life_rewards = np.mean(rewards)
                # Write summaries
                self.model.write_summaries(self.sess, i_step, b_s, b_s_, b_a, b_r, b_d)
                self.model.summary_scalar(self.sess, i_step, 'epsilon', epsilon)
                self.model.summary_scalar(self.sess, i_step, 'learning_rate', lr)
                self.model.summary_scalar(self.sess, i_step, 'steps/sec', steps_sec)
                self.model.summary_scalar(self.sess, i_step, 'reward_by_life',
                                          mean_life_rewards)
                self.model.summary_scalar(self.sess, i_step, 'reward_by_episode(unclipped)',
                                          mean_ep_rewards)
                # Format data
                header = 'Step {}/{} ({:.2f}%) ETA: {}'.format(
                    i_step, int(num_steps), 100 * i_step / num_steps, eta_formatted)
                tags = ['Reward Mean [{} episodes](unclipped)'.format(num_new_episodes),
                        'Reward Mean [{} lives]'.format(len(rewards)),
                        'Learning Rate',
                        'Exploration Rate',
                        'Steps/Seconds']
                values = ['{:.2f}'.format(mean_ep_rewards),
                          '{:.2f}'.format(mean_life_rewards),
                          '{:.5f}'.format(lr),
                          '{:.3f}'.format(epsilon),
                          '{:.2f}'.format(steps_sec)]
                print_table(tags, values, header=header)

                # Reset rewards
                rewards = []
