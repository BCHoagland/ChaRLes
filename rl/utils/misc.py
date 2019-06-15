import numpy as np

def noisy_action(a, std, env, clip=None):
    noise = np.random.normal(0, std)
    if clip is not None:
        noise = np.clip(noise, -clip, clip)
    return np.clip(a.numpy() + noise, env.action_space.low, env.action_space.high)
