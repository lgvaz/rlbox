import gym
from gymmeforce.agents import DQNAgent
from gymmeforce.common.utils import piecewise_linear_decay


# Create gym enviroment
env_name = 'LunarLander-v2'

# Define learning rate and exploration schedule
num_steps = 6e5
learning_rate_schedule = piecewise_linear_decay(boundaries=[0.1 * num_steps, 0.5 * num_steps],
                                                values=[1, .1, .1],
                                                initial_value=1e-3)
exploration_schedule = piecewise_linear_decay(boundaries=[0.1 * num_steps, 0.5 * num_steps],
                                              values=[.1, .01, .01],
                                              initial_value=1.)

# Create agent
agent = DQNAgent(env_name=env_name,
                 log_dir='logs/lunar_lander/dueling_ddqn_v4',
                 history_length=1,
                 double=True,
                 dueling=True)
# Train
agent.train(num_steps=num_steps,
            learning_rate=learning_rate_schedule,
            exploration_schedule=exploration_schedule,
            replay_buffer_size=2e4,
            target_update_freq=1000)
