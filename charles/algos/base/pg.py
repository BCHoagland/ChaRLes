from charles.algos.algorithm import Algorithm
from charles.models import *

class PG(Algorithm):
    def __init__(self):
        self.name = 'PG'
        self.type = 'on-policy'
        self.color = [0, 85, 135]

    def setup(self):
        self.π = Model(LinearPolicy, self.env, self.config.lr)

    def interact(self, s):
        a = self.π(s)
        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, m = storage.get_all()

        # calculate returns
        returns = [0] * len(r)
        discounted_next = 0
        for i in reversed(range(len(r))):
            returns[i] = r[i] + discounted_next
            discounted_next = 0.99 * returns[i] * m[i - 1]

        # normalize returns
        returns = torch.stack(returns)
        mean = returns.mean()
        std = returns.std()
        returns = (returns - mean) / (std + 1e-6)

        # calculate log probabilities
        log_p = self.π.log_prob(s, a)

        # update policy
        loss = -(returns * log_p).mean()
        self.π.optimize(loss)
