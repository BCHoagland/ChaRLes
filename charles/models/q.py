import torch
import torch.nn as nn
from charles.models.base import Network

class Q(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        n_h = 64

        self.pre_state = nn.Sequential(
            nn.Linear(self.n_obs, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h // 2),
            nn.ELU()
        )

        self.pre_action = nn.Sequential(
            nn.Linear(self.n_acts, n_h // 2),
            nn.ELU()
        )

        self.main = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, 1)
        )

    def forward(self, s, a):
        s = self.pre_state(s)
        a = self.pre_action(a)
        return self.main(torch.cat([s, a], -1))

class V(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.main(s)

class DQNNet(Network):
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
