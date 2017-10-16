from gymmeforce.agents import PPOAgent

env_name = 'LunarLander-v2'
# log_dir = 'logs/walker2d/ppo/3e4lr_02clip_00entropy_na_ub_nb_400len_5000batch_10epochs_v0'
log_dir = 'logs/lunar_lander/ppo/1e3lr_02clip_00entropy_na_ub_nb_2000batch_10epochs_v0'

agent = PPOAgent(env_name,
                 log_dir=log_dir,
                 epsilon_clip=0.2,
                 entropy_coef=0.0,
                 normalize_advantages=True,
                 use_baseline=True,
                 normalize_baseline=True)

agent.train(1e-3,
            # max_steps=1e7,
            max_iters=100,
            # max_episode_steps=400,
            timesteps_per_batch=2000,
            num_epochs=10,
            batch_size=64,
            record_freq=None)


# agent = PPOAgent('InvertedDoublePendulum-v1',
#                        # log_dir='logs/lunar_lander/ppo/na_baseline_nb_entropy0_2000batch_10epochs_v2',
#                  log_dir='logs/inverted_double_pendulum/5e4lr_02clip_na_baseline_nb_entropy0_3000batch_10epochs_v0',
#                  entropy_coef=0.0,
#                  normalize_advantages=True,
#                  use_baseline=True,
#                  normalize_baseline=False,
#                  debug=True)

# agent.train(3e-4,
#             max_iters=70,
#             timesteps_per_batch=3000,
#             max_episode_steps=1000,
#             num_epochs=10,
#             record_freq=None)
