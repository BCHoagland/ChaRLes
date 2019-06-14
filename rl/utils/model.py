import torch
from copy import deepcopy

class Model():
    def __init__(self, model, lr, target=False, τ=0.995, optim=torch.optim.Adam):
        self.model = model
        if target: self.target_model = deepcopy(model)
        self.τ = τ
        self.optimizer = optim(self.model.parameters(), lr=lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target(self, *args):
        return self.target_model(*args)

    def __getattr__(self, k):
        return getattr(self.model, k)

    def __call__(self, *args):
        return self.model(*args)

    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((self.τ * target_param.data) + ((1 - self.τ) * param.data))
