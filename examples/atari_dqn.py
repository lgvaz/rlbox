import gym
from gymmeforce.agents import DQNAgent
from gymmeforce.wrappers import wrap_deepmind
from gymmeforce.common.utils import piecewise_linear_decay


# Create gym enviroment
env = gym.make('SpaceInvadersNoFrameskip-v4')

# Define learning rate and exploration schedule
num_steps = 40e6
learning_rate_schedule = piecewise_linear_decay(boundaries=[0.1 * num_steps, 0.5 * num_steps],
                                                values=[1, .5, .5],
                                                initial_value=1e-4)
exploration_rate_schedule = piecewise_linear_decay(boundaries=[1e6, 0.5 * num_steps],
                                                   values=[.1, .01, .01],
                                                   initial_value=1.)

# Create agent
agent = DQNAgent(env=env,
                 log_dir='logs/space_invaders/v0',
                 double=True,
                 env_wrapper=wrap_deepmind)
# Train
agent.train(num_steps=num_steps,
            learning_rate=learning_rate_schedule,
            exploration_schedule=exploration_rate_schedule,
            replay_buffer_size=1e6,
            target_update_freq=30000)
