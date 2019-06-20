import random
import numpy as np
from charles.algos.algorithm import Algorithm
from charles.models import *

class DQfD(Algorithm):
    def __init__(self):
        self.name = 'DQfD'
        self.type = 'off-policy'
        self.color = [180, 0, 0]

    def setup(self):
        self.Q = Model(DQNNet, self.env, self.config.lr, target=True)

    def interact(self, s):
        if random.random() < 0.05:
            a = self.random_action()
        else:
            a = np.argmax(self.Q(s), axis=1)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def l(self, a_e):
        return (1 - torch.eye(self.env.action_space.n))[a_e.squeeze().long()]

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        # DQN loss
        max_next_q, _ = torch.max(self.Q.target(s2), dim=2, keepdim=True)
        y = r + (0.99 * m * max_next_q)
        dq_loss = torch.pow(self.Q(s).gather(2, a.long()) - y, 2).mean()

        # large margin classification loss
        max_q, _ = torch.max(self.Q(s) + self.l(a), dim=2, keepdim=True)
        e_loss = (max_q - self.Q(s).gather(2, a.long())).mean()

        # optimization step and target network update
        loss = dq_loss + e_loss
        self.Q.optimize(loss)
        self.Q.soft_update_target()

        return loss
