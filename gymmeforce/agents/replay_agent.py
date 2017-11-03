from gymmeforce.agents.base_agent import BaseAgent
from gymmeforce.common.utils import ReplayBuffer, RingBuffer


class ReplayAgent(BaseAgent):
    def __init__(self, env_name, history_length=4, **kwargs):
        super().__init__(env_name, **kwargs)
        self.history_length = history_length
        self.replay_buffer = None
        # Keep track of past states
        self.states_history = RingBuffer(self.env_config['state_shape'], history_length)
        self.env_config['state_shape'] += (self.history_length, )

    def _play_and_add_to_buffer(self, ep_runner):
        trajectory = ep_runner.run_one_step(self.select_action)
        # Store experience
        self.replay_buffer.add(trajectory['state'], trajectory['action'],
                               trajectory['reward'], trajectory['done'])

        return trajectory

    def _populate_replay_buffer(self, ep_runner, replay_buffer_size, init_buffer_size,
                                batch_size, n_step):
        # Create replay buffer
        if self.replay_buffer is None:
            print('Creating replay buffer')
            self.replay_buffer = ReplayBuffer(
                int(replay_buffer_size),
                history_length=self.history_length,
                batch_size=batch_size,
                n_step=n_step)

            # Populate replay buffer with random agent
            num_init_replays = replay_buffer_size * init_buffer_size
            self.epsilon = 1
            for i in range(int(num_init_replays)):
                self._play_and_add_to_buffer(ep_runner)
                # Logs
                if i % 100 == 0:
                    print(
                        '\rPopulating replay buffer: {:.1f}%'.format(
                            i * 100 / num_init_replays),
                        end='',
                        flush=True)

        print('\rPopulating replay buffer: DONE!')
