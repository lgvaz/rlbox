from rlbox.agents import VanillaPGAgent

agent = VanillaPGAgent(
    'CartPole-v0',
    log_dir='tests/cart_pole/advantages/na_ub_nb_v1',
    entropy_coef=0.1,
    normalize_advantages=True,
    use_baseline=True)

agent.train(
    5e-3,
    gamma=0.99,
    max_iters=100,
    timesteps_per_batch=2000,
    num_epochs=1,
    record_freq=None)
