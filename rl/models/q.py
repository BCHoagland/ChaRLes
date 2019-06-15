import torch
import torch.nn as nn
from rl.models.base import Network

class Q(Network):
    def __init__(self, env):
        super().__init__(env)

        self.pre_state = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )

        self.pre_action = nn.Sequential(
            nn.Linear(self.n_acts, 32),
            nn.ELU()
        )

        self.main = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        s = self.pre_state(s)
        a = self.pre_action(a)
        return self.main(torch.cat([s, a], 1))

class V(Network):
    def __init__(self, env):
        super().__init__(env)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.main(s)
