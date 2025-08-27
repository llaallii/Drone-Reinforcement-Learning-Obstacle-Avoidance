import gymnasium as gym

class BaseEnv(gym.Env):
    """Common interface for drone navigation environments."""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
