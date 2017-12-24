import gym
from gymmeforce.agents import DQNAgent
from gymmeforce.common.schedules import piecewise_linear_decay

# Create gym enviroment
env_name = 'LunarLander-v2'

# Define learning rate and exploration schedule
max_steps = 1e5
learning_rate_schedule = piecewise_linear_decay(
    boundaries=[0.1 * max_steps, 0.5 * max_steps], values=[1, .1, .1], initial_value=1e-3)
exploration_rate = piecewise_linear_decay(
    boundaries=[0.1 * max_steps, 0.5 * max_steps],
    values=[.1, .01, .01],
    initial_value=1.)

# Create agent
agent = DQNAgent(
    env_name=env_name,
    log_dir='logs/lunar_lander/random_10n_step_uncareful_sample_softtarget_v0_12',
    history_length=1,
    double=True,
    dueling=True,
    target_update_freq=10,
    target_soft_update=0.01)
# Train
agent.train(
    max_steps=max_steps,
    n_step=4,
    randomize_n_step=True,
    learning_rate=learning_rate_schedule,
    exploration_rate=exploration_rate,
    replay_buffer_size=2e4,
    log_steps=1e4)
