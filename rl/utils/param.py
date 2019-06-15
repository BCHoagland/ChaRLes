import numpy as np
import torch

class LearnableParam():
    def __init__(self, init_value, lr, optim=torch.optim.Adam):
        self.log = torch.tensor(np.log(init_value), requires_grad=True)
        self.optimizer = optim([self.log], lr=lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def value(self):
        return self.log.exp()

    def __getattr__(self, k):
        return getattr(self.value(), k)

    def __mul__(self, other):
        return self.value() * other
