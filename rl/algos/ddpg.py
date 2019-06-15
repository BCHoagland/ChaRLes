from rl.algorithm import Algorithm
from rl.models import *
from rl.utils import *

class DDPG(Algorithm):
    def setup(self):
        self.name = 'DDPG'
        self.type = 'off-policy'

        self.μ = Model(DeterministicPolicy(self.env), 1e-3, target=True)
        self.Q = Model(Q(self.env), 1e-4, target=True)

        self.explore()

    def interact(self, s):
        a = noisy_action(self.μ(s), 0.15, self.env)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        y = r + (0.99 * m * self.Q.target(s2, self.μ.target(s2)))
        q_loss = torch.pow(self.Q(s, a) - y, 2).mean()
        self.Q.optimize(q_loss)

        policy_loss = -self.Q(s, self.μ(s)).mean()
        self.μ.optimize(policy_loss)

        self.μ.soft_update_target()
        self.Q.soft_update_target()
