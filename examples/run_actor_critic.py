from rlbox.agents import ActorCriticAgent

agent = ActorCriticAgent(
    'InvertedPendulum-v1',
    log_dir='tests/inverted_pendulum/advantages/na_ub_nb_v1',
    entropy_coef=0.0,
    normalize_advantages=True)

agent.train(
    5e-3,
    use_gae=True,
    gamma=0.99,
    gae_lambda=0.95,
    max_iters=100,
    timesteps_per_batch=2000,
    num_epochs=1,
    record_freq=None)
