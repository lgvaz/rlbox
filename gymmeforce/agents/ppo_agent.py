from gymmeforce.agents import ActorCriticAgent
from gymmeforce.models import PPOModel


class PPOAgent(ActorCriticAgent):
    '''
    Proximal Policy Optimization as described in (https://arxiv.org/pdf/1707.06347.pdf)
    PPO is implemented in two main methods:
    	* Clipped Surrogate Objective: The probability ratio between the new and old
    	  policy is clipped in the range [1 - epsilon_clip, 1 + epsilon_clip],
    	  this limits the change in the policy that each update can make

    	* Adaptive KL Penalty Coefficient: A penalty based on the KL divergence betweeen
    	  the old and new policy is added to the loss, the penalty coefficient is
    	  automatically adapted to achieve some target KL divergence value.

    Args:
    	env_name: Gym environment name

    Keyword args:
        ppo_clip: Whether or not to use the Clipped Surrogate Objective (default True)
        ppo_adaptive_kl: Whether or not to use the Adaptive KL Penalty Coefficient
            (default False)
    	epsilon_clip: Probability ratio clipping (default 0.2)
        kl_coef: Starting KL penalty coefficient (default 1.)
        kl_targ: KL target between old and new policy on each update (default 0.01)
        kl_hinge_coef: Multiplier for the hinge loss on the KL divergence (default 50)
        normalize_advantages: Whether or not to normalize advantages (default False)
        use_baseline: Whether or not to subtract a baseline(NN representing the
            value function) from the returns (default True)
        entropy_coef: Entropy penalty added to the loss (default 0.0)
        policy_graph: Function returning a tensorflow graph representing the policy
            (default None)
        value_graph: Function returning a tensorflow graph representing the value function
            (default None)
        log_dir: Directory used for writing logs (default 'logs/examples')
    '''

    def _create_model(self, **kwargs):
        return PPOModel(self.env_config, **kwargs)

    def generate_batch(self, **kwargs):
        super().generate_batch(**kwargs)
        self.batch['ppo_clip_range'] = self._calculate_schedule(self.ppo_clip_range)

    def train(self, ppo_clip_range=0.2, **kwargs):
        self.ppo_clip_range = ppo_clip_range
        super().train(**kwargs)
