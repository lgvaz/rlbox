from gymmeforce.agents import PPOAgent

env_name = 'Hopper-v1'
# log_dir = 'logs/walker2d/ppo/3e4lr_02clip_00entropy_na_ub_nb_400len_5000batch_10epochs_v0'
# log_dir = 'logs/lunar_lander/ppo/1e3lr_02clip_00entropy_na_ub_nb_2000batch_10epochs_v0'
log_dir = 'tests/hopper/ppo/3e4lr_02clip_00entropy_na_0.95lambda_2000batch_10epochs_v9'
# log_dir = 'tests/cheetah/3e-4lr_02clip_00entropy_na_095lambda_400len_2000batch_10epochs_new_vtarg_v0'

agent = PPOAgent(env_name,
                 log_dir=log_dir,
                 epsilon_clip=0.2,
                 entropy_coef=0.0,
                 normalize_advantages=True)

agent.train(3e-4,
            # max_steps=1e7,
            max_iters=1e6,
            # max_episode_steps=400,
            timesteps_per_batch=2000,
            gamma=0.99,
            gae_lambda=0.95,
            num_epochs=10,
            batch_size=64,
            record_freq=None)


# agent = PPOAgent('InvertedDoublePendulum-v1',
#                        # log_dir='logs/lunar_lander/ppo/na_baseline_nb_entropy0_2000batch_10epochs_v2',
#                  log_dir='tests/inverted_double_pendulum/ppo_actor_critic_gae_vtarget_corrected2',
#                  entropy_coef=0.0,
#                  normalize_advantages=True)

# agent.train(3e-4,
#             max_iters=100,
#             use_gae=True,
#             gamma=0.99,
#             gae_lambda=0.95,
#             timesteps_per_batch=3000,
#             num_epochs=10,
#             record_freq=None)
