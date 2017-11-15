import gym

from gymmeforce.agents import DQNAgent
from gymmeforce.common.utils import piecewise_linear_decay
from gymmeforce.wrappers import wrap_deepmind

# Create gym enviroment
env_name = 'SpaceInvadersNoFrameskip-v4'
# env_name = 'PongNoFrameskip-v4'

# Define learning rate and exploration schedule
max_steps = 40e6
learning_rate_schedule = piecewise_linear_decay(
    boundaries=[0.1 * max_steps, 0.5 * max_steps], values=[1, .5, .5], initial_value=1e-4)
exploration_schedule = piecewise_linear_decay(
    boundaries=[1e6, 0.5 * max_steps], values=[.1, .01, .01], initial_value=1.)

# Create agent
agent = DQNAgent(
    env_name=env_name,
    log_dir='logs/space_invaders/40M_random_20n_step_dueling_ddqn_30ktarget_v0_0',
    double=True,
    dueling=True,
    target_update_freq=30000,
    env_wrapper=wrap_deepmind)
# Train
agent.train(
    max_steps=max_steps,
    n_step=20,
    randomize_n_step=True,
    learning_rate=learning_rate_schedule,
    exploration_schedule=exploration_schedule,
    replay_buffer_size=1e6,
    log_steps=4e4)
