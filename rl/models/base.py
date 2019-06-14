import torch.nn as nn

class Network(nn.Module):
    def __init__(self, env):
        super(Network, self).__init__()

        self.env = env

        self.n_obs = env.observation_space.shape[0]
        try:
            self.n_acts = env.action_space.shape[0]
        except:
            self.n_acts = env.action_space.n

        try:
            self.min_a = env.action_space.high[0]
            self.max_a = env.action_space.high[0]
        except:
            self.min_a = None
            self.max_a = None
