import time
import numpy as np
import tensorflow as tf
from gymmeforce.common.utils import discounted_sum_rewards_final_sum
from gymmeforce.models import DQNModel
from gymmeforce.common.gym_utils import EpisodeRunner
from gymmeforce.agents import ReplayAgent


class DQNAgent(ReplayAgent):
    def __init__(self, env_name, **kwargs):
        super().__init__(env_name, **kwargs)
        self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        self.model = DQNModel(self.env_config, **kwargs)

    def select_action(self, state):
        # Concatenates <history_length> states
        self.states_history.append(state)
        state_hist = self.states_history.get_data()

        # Select action based on an egreedy policy
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.env_config['num_actions'])
        else:
            Q_values = self.model.predict(self.sess, state_hist[np.newaxis])
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
              init_buffer_size=0.05, batch_size=32, record_freq=None, log_steps=2e4):
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
        monitored_env, env = self._create_env('videos/train', record_freq)
        ep_runner = EpisodeRunner(env)

        self._populate_replay_buffer(ep_runner, replay_buffer_size, init_buffer_size, batch_size, n_step)
        # Create training ops
        self.model.create_training_ops(gamma, clip_norm, target_soft_update)
        # Create Session
        self._maybe_create_tf_sess()
        self.logger.add_tf_writer(self.sess, self.model.summary_scalar)

        print('Started training')
        num_episodes = 0
        reward_sum = 0
        start_time = time.time()
        # TODO: soft updating here, need to hard copy weights
        self.model.update_target_net(self.sess)
        for i_step in range(1, int(num_steps) + 1):
            self.epsilon = exploration_schedule(i_step)
            trajectory = self._play_and_add_to_buffer(ep_runner)
            reward_sum += trajectory['reward']

            if trajectory['done']:
                self.logger.add_log('Reward/Life', reward_sum)
                reward_sum = 0

            # Perform gradient descent
            if i_step % learning_freq == 0:
                # Get batch to train on
                random_n_step = np.random.randint(1, n_step)
                b_s, b_s_, b_a, b_r, b_d = self.replay_buffer.sample(random_n_step)
                # Calculate n_step rewards
                b_r = [discounted_sum_rewards_final_sum(r, d) for r, d in zip(b_r, b_d)]
                b_d = np.any(b_d, axis=1)
                # Calculate learning rate
                if callable(learning_rate):
                    lr = learning_rate(i_step)
                else:
                    lr = learning_rate
                self.model.fit(self.sess, lr, b_s, b_s_, b_a, b_r, b_d, random_n_step)

            # Update target network
            if i_step % target_update_freq == 0:
                self.model.update_target_net(self.sess)

            if i_step % log_steps == 0:
                self.model.increase_global_step(self.sess, log_steps)
                # Save model
                # self.model.save(self.sess, i_step)
                # Calculate rewards statistics
                ep_rewards = monitored_env.get_episode_rewards()
                num_episodes_old = num_episodes
                num_episodes = len(ep_rewards)
                num_new_episodes = num_episodes - num_episodes_old
                mean_ep_rewards = np.mean(ep_rewards[-num_new_episodes:])
                # Write summaries
                self.model.write_summaries(self.sess, i_step, b_s, b_s_, b_a, b_r, b_d, random_n_step)
                # Write logs
                self.logger.add_log('Reward/Episode (unclipped)', mean_ep_rewards)
                self.logger.add_log('Learning Rate', lr, precision=5)
                self.logger.add_log('Exploration Rate', self.epsilon, precision=3)
                self.logger.timeit(log_steps, max_steps=num_steps)
                self.logger.log('Step {}/{} ({:.2f}%)'.format(
                    i_step, int(num_steps), 100 * i_step / num_steps))
