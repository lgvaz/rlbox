from gymmeforce.agents import DQNAgent
from gymmeforce.models import CategoricalDQNModel


class CategoricalDQNAgent(DQNAgent):
    def _create_model(self, **kwargs):
        self.model = CategoricalDQNModel(self.env_config, **kwargs)
