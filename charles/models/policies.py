import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from charles.models.base import Network

class LinearPolicy:
    def __init__(self, env, n_obs=None):

        if env.action_space.__class__.__name__ == 'Discrete':
            self.net = CategoricalPolicy(env, n_obs)
        else:
            self.net = StochasticPolicy(env, n_obs)

    def __getattr__(self, k):
        return getattr(self.net, k)

    def __call__(self, *args):
        return self.net(*args)

class CategoricalPolicy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

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
        return a

    def log_prob(self, s, a):
        orig_s_shape = s.shape
        s = s.squeeze()
        a = a.squeeze()
        if len(s.shape) == 0:
            s = s.unsqueeze(0)
        if len(a.shape) == 0:
            a = a.unsqueeze(0)

        log_p = self.dist(s).log_prob(a)

        while len(orig_s_shape) > len(log_p.shape):
            log_p = log_p.unsqueeze(len(log_p.shape))

        return log_p

class StochasticPolicy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

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
        return a

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)

class DeterministicPolicy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

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
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

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
        return a, log_p
