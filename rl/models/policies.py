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
        )

    def dist(self, s):
        logits = self.mean(s)
        return Categorical(logits=logits)

    def forward(self, s):
        a = self.dist(s).sample()
        return a.numpy()

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)

class StochasticPolicy(Network):
    def __init__(self, env):
        super(StochasticPolicy, self).__init__(env)

        self.mean = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(self.n_acts))

    def dist(self, s):
        mean = self.mean(torch.FloatTensor(s))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, s):
        a = self.dist(s).sample()
        a = ((a + 1) / 2) * (self.max_a - self.min_a) + self.min_a
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

class TanhPolicy(Network):
    def __init__(self, env):
        super(TanhPolicy, self).__init__(env)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(64, self.n_acts)
        )

        self.log_std = nn.Sequential(
            nn.Linear(64, self.n_acts)
        )

    def dist(self, s):
        s = self.main(s)
        mean = self.mean(s)
        std = torch.exp(self.log_std(s).expand_as(mean))
        dist = Normal(mean, std)
        return dist

    def forward(self, s):
        dist = self.dist(s)
        a = torch.tanh(dist.sample())
        return a

    def sample(self, s):
        dist = self.dist(s)
        x = dist.rsample()
        a = torch.tanh(x)
        log_p = dist.log_prob(x)
        log_p -= torch.log(1 - torch.pow(a, 2) + 1e-6)
        a = torch.pow(a, 1)
        return a, log_p
