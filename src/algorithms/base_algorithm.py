class BaseAlgorithm:
    def __init__(self, env_fn, config=None):
        self.env_fn = env_fn
        self.config = config or {}

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
