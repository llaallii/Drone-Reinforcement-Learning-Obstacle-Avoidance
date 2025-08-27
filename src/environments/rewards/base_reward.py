class BaseReward:
    def __call__(self, env, state, action, next_state):
        raise NotImplementedError
