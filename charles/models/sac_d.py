import torch
import torch.nn as nn
from torch.distributions import Categorical
from charles.models.base import Network

class d_Q(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts)
        )

    def forward(self, s):
        return self.main(s)

class d_Policy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            nn.Softmax(dim=-1)
        )

    def forward(self, s):
        s = torch.FloatTensor(s)
        return self.main(s)
    
    def select_action(self, s):
        a = self.forward(s)
        dist = Categorical(a)
        return dist.sample()
    
    def log_probs(self, s):
        log_p = torch.log(self.main(s))
        return log_p