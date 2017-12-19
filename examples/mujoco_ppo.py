from gymmeforce.agents import PPOAgent

env_name = 'HalfCheetah-v1'
# log_dir = 'tests/hopper/ppo/adaptive_kl_50eta_earlystop_v0'
# log_dir = 'logs/cheetah/ppo/baselinenet_adaptive_kl_scale_1000timesteps_v8'
log_dir = 'logs/cheetah/ppo/baselinenet_clip_kl_scale_1000timesteps_v1'
# log_dir = 'logs/lunar_lander/ppo/1e3lr_02clip_00entropy_na_ub_nb_2000batch_10epochs_v0'
# log_dir = 'tests/hopper/ppo/3e4lr_02clip_00entropy_na_0.95lambda_2000batch_10epochs_v9'
# log_dir = 'tests/cheetah/3e-4lr_02clip_00entropy_na_095lambda_400len_2000batch_10epochs_new_vtarg_v0'

agent = PPOAgent(
    env_name,
    ppo_clip=True,
    ppo_adaptive_kl=False,
    scale_states=True,
    log_dir=log_dir,
    epsilon_clip=0.2,
    entropy_coef=0.0,
    normalize_advantages=True)

agent.train(
    3e-4,
    # max_steps=1e6,
    # max_iters=1e6,
    max_episodes=3000,
    # max_episode_steps=400,
    timesteps_per_batch=5000,
    gamma=0.99,
    gae_lambda=0.95,
    num_epochs=10,
    batch_size=64,
    record_freq=50)

# agent = PPOAgent(
#     'InvertedDoublePendulum-v1',
#     # log_dir='logs/lunar_lander/ppo/na_baseline_nb_entropy0_2000batch_10epochs_v2',
#     log_dir='logs/inverted_double_pendulum/ppo/adaptive_kl_betaeta_v0',
#     entropy_coef=0.0,
#     normalize_advantages=True)

# agent.train(
#     3e-4,
#     # max_iters=100,
#     max_steps=5e5,
#     use_gae=True,
#     gamma=0.99,
#     gae_lambda=0.95,
#     timesteps_per_batch=3000,
#     num_epochs=10,
#     record_freq=None)
