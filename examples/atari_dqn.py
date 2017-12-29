import gym

from rlbox.agents import DQNAgent
from rlbox.common.schedules import piecewise_linear_decay
from rlbox.wrappers import AtariWrapper

# Create gym enviroment
env_name = 'SpaceInvadersNoFrameskip-v4'

# Define wrapper
frame_skip = 4
env_wrapper = AtariWrapper(frame_skip=frame_skip)
# Define learning rate and exploration schedule
max_steps = 40e6 * frame_skip
learning_rate_schedule = piecewise_linear_decay(
    boundaries=[0.1 * max_steps, 0.5 * max_steps], values=[1, .5, .5], initial_value=1e-4)
exploration_schedule = piecewise_linear_decay(
    boundaries=[1e6, 0.5 * max_steps], values=[.1, .01, .01], initial_value=1.)

# Create agent
agent = DQNAgent(
    env_name=env_name,
    log_dir='logs/space_invaders/40M_random_4n_step_dueling_ddqn_30ktarget_v0',
    double=True,
    dueling=True,
    target_update_freq=30000,
    env_wrapper=env_wrapper)
# Train
agent.train(
    max_steps=max_steps,
    n_step=4,
    randomize_n_step=True,
    learning_rate=learning_rate_schedule,
    exploration_rate=exploration_schedule,
    replay_buffer_size=1e6,
    log_steps=4e4)
