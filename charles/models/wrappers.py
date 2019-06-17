import numpy as np
import torch

class Model:
    def __init__(self, model_type, env, lr, target=False, τ=0.995, optim=torch.optim.Adam):
        self.model = model_type(env)
        if target:
            self.target_model = model_type(env)
            self.target_model.load_state_dict(self.model.state_dict())
        self.τ = τ
        self.optimizer = optim(self.model.parameters(), lr=lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target(self, *args):
        with torch.no_grad():
            return self.target_model(*[torch.FloatTensor(arg) for arg in args])

    def log_prob(self, *args):
        return self.model.log_prob(*[torch.FloatTensor(arg) for arg in args])

    def __getattr__(self, k):
        return getattr(self.model, k)

    def __call__(self, *args):
        return self.model(*[torch.FloatTensor(arg) for arg in args])

    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((self.τ * target_param.data) + ((1 - self.τ) * param.data))

class LearnableParam:
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
