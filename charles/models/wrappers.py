import numpy as np
import torch

class Model:
    def __init__(self, model_type, env, lr, n_obs=None, target=False, τ=0.995, optim=torch.optim.Adam):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = model_type(env, n_obs).to(self.device)
        if target:
            self.target_model = model_type(env, n_obs).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
        self.τ = τ
        self.optimizer = optim(self.model.parameters(), lr=lr)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target(self, *args):
        with torch.no_grad():
            try:
                return self.target_model(*[torch.FloatTensor(arg).to(self.device) for arg in args])
            except:
                return self.target_model(*args)

    def log_prob(self, *args):
        try:
            return self.model.log_prob(*[torch.FloatTensor(arg).to(self.device) for arg in args])
        except:
            return self.model.log_prob(*args)

    def __getattr__(self, k):
        return getattr(self.model, k)

    def __call__(self, *args):
        try:
            return self.model(*args)
        except:
            return self.model(*[torch.FloatTensor(arg).to(self.device) for arg in args])

    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((self.τ * target_param.data) + ((1 - self.τ) * param.data))

class LearnableParam:
    def __init__(self, init_value, lr, optim=torch.optim.Adam):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.log = torch.tensor(np.log(init_value), requires_grad=True, device=self.device)
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
