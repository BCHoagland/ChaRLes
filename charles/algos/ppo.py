from charles.algorithm import Algorithm
from charles.models import *
from charles.utils import *

class PPO(Algorithm):
    def setup(self):
        self.name = 'PPO'
        self.type = 'on-policy'
        self.color = [153, 51, 153]

        self.π = Model(StochasticPolicy, self.env, 1e-3)
        self.V = Model(V, self.env, 1e-3)

    def interact(self, s):
        a = self.π(s)
        log_p = self.π.log_prob(s, a)
        v = self.V(s)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, v, log_p, done)
        return s2, r, done, data

    def update(self, storage):
        for s, a, r, v, old_log_p, m in storage.get_batches():

            # calculate returns
            returns = [0] * len(r)
            discounted_next = 0
            for i in reversed(range(len(r))):
                returns[i] = r[i] + discounted_next
                discounted_next = 0.99 * returns[i] * m[i - 1]
            returns = torch.stack(returns)

            # calculate and normalize advantage
            adv = returns - v
            mean = adv.mean()
            std = adv.std()
            adv = (adv - mean) / (std + 1e-6)

            # calculate new log probabilities
            new_log_p = self.π.log_prob(s, a)

            ratio = torch.exp(new_log_p - old_log_p)
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv)

            policy_loss = -surrogate.mean()
            self.π.optimize(policy_loss)

            value_loss = torch.pow(self.V(s) - returns, 2).mean()
            self.V.optimize(value_loss)
