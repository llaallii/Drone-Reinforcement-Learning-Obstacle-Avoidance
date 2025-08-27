from .base_env import BaseEnv
from .procedural_forest import ProceduralForest

class ForestEnv(BaseEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.world = ProceduralForest(config or {})

    def step(self, action):
        # TODO: implement environment dynamics and rewards
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        # TODO: implement reset logic
        raise NotImplementedError
