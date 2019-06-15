import numpy as np

def clip_action(a, noise, env):
    return np.clip(a.numpy() + noise, env.action_space.low, env.action_space.high)
