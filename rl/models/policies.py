import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from rl.models.base import Network

class CategoricalPolicy(Network):
    def __init__(self, env):
        super(CategoricalPolicy, self).__init__(env)

        self.mean = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            # nn.Tanh()
        )

        # self.log_std = nn.Parameter(torch.zeros(self.n_acts))

    def dist(self, s):
        # mean = self.mean(torch.FloatTensor(s))
        # std = self.log_std.exp().expand_as(mean)
        # return Normal(mean, std)
        logits = self.mean(s)
        return Categorical(logits=logits)

    def forward(self, s):
        a = self.dist(s).sample()
        # a = ((a + 1) / 2) * (self.max_a - self.min_a) + self.min_a
        return a.numpy()

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)

class DeterministicPolicy(Network):
    def __init__(self, env):
        super(DeterministicPolicy, self).__init__(env)

        self.mean = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            nn.Tanh()
        )

    def forward(self, s):
        a = self.mean(s)
        a = ((a + 1) / 2) * (self.max_a - self.min_a) + self.min_a
        return a
