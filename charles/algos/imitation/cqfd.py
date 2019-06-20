import random
import numpy as np
from math import sqrt, pi
from charles.algos.algorithm import Algorithm
from charles.models import *

class CQfD(Algorithm):
    def __init__(self):
        self.name = 'CQfD-new'
        self.type = 'off-policy'
        # self.color = [0, 0, 180]
        self.color = [0, 0, 255]

    def setup(self):
        self.μ = Model(DeterministicPolicy, self.env, self.config.lr, target=True)
        self.Q = Model(Q, self.env, self.config.lr, target=True)

    def interact(self, s):
        a = self.noisy_action(self.μ(s), 0.15)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def l(self, a, a_e):
        return (1 - torch.exp(-torch.pow(a - a_e, 2))) / sqrt(2 * pi)

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        y = r + (0.99 * m * self.Q.target(s2, self.μ.target(s2)))
        mse_loss = torch.pow(self.Q(s, a) - y, 2).mean()
        e_loss = (self.Q(s, self.μ(s)) + self.l(self.μ(s), a) - self.Q(s, a)).mean()
        # e_loss = 0

        q_loss = mse_loss + e_loss
        self.Q.optimize(q_loss)

        # policy_loss = -self.Q(s, self.μ(s)).mean()
        policy_loss = torch.pow(a - self.μ(s), 2).mean()
        self.μ.optimize(policy_loss)

        self.μ.soft_update_target()
        self.Q.soft_update_target()

        return q_loss
