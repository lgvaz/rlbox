from gymmeforce.agents import PPOAgent


agent = PPOAgent('HalfCheetah-v1',
                 log_dir='tests/ppo/cheetah/v0',
                 epsilon_clip=0.2,
                 entropy_coef=0.0,
                 normalize_advantages=False,
                 use_baseline=True,
                 normalize_baseline=False)

agent.train(5e-4,
            max_steps=1e6,
            max_episode_steps=500,
            timesteps_per_batch=3000,
            num_epochs=10,
            batch_size=64,
            record_freq=100)
