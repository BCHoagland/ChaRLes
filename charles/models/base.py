import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.env = env

        try:
            self.n_obs = env.observation_space.shape[0]
        except:
            self.n_obs = 1

        try:
            self.n_acts = env.action_space.shape[0]
        except:
            self.n_acts = env.action_space.n

        try:
            self.min_a = env.action_space.low
            self.max_a = env.action_space.high
        except:
            self.min_a = None
            self.max_a = None
