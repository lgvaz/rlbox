from gymmeforce.agents import PPOAgent
from gymmeforce.common.schedules import piecewise_linear_decay

env_name = 'Hopper-v1'
log_dir = 'logs_bench/hopper/ppo/02clip_v2_0'

max_steps = 1e6
# Optionally you can define schedules
# learning_rate_schedule = piecewise_linear_decay(
#     boundaries=[0.2 * max_steps, 0.6 * max_steps], values=[1, .5, .5], initial_value=3e-4)
# clip_range_schedule = piecewise_linear_decay(
#     boundaries=[0.1 * max_steps, 0.8 * max_steps], values=[1, .1, .1], initial_value=0.2)

agent = PPOAgent(
    env_name,
    ppo_clip=True,
    ppo_adaptive_kl=False,
    kl_targ=0.01,
    # hinge_coef=1000,
    scale_states=True,
    # grad_clip_norm=10,
    log_dir=log_dir,
    normalize_advantages=True)

agent.train(
    # learning_rate=learning_rate_schedule,
    learning_rate=3e-4,
    ppo_clip_range=0.2,
    max_steps=max_steps,
    # max_iters=1e6,
    # max_episodes=30000,
    # max_episode_steps=400,
    timesteps_per_batch=2048,
    # episodes_per_batch=5,
    # use_gae=False,
    gamma=0.995,
    gae_lambda=0.97,
    num_epochs=10,
    batch_size=64,
    record_freq=400)
