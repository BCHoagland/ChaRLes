from rl.algorithm import Algorithm
from rl.models import *
from rl.utils import *

class TD3(Algorithm):
    def setup(self):
        self.name = 'TD3'

        self.μ = Model(DeterministicPolicy(self.env), 1e-3, target=True)
        self.Q1 = Model(Q(self.env), 1e-4, target=True)
        self.Q2 = Model(Q(self.env), 1e-4, target=True)

        self.updates = 0

        self.explore()

    def interact(self, s):
        a = noisy_action(self.μ(s), 0.15, self.env)

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        a2 = noisy_action(self.μ.target(s2), 0.15, self.env, clip=0.5)
        min_next_q = torch.min(self.Q1.target(s2, a2), self.Q2.target(s2, a2))
        y = r + (0.99 * m * min_next_q)

        q1_loss = torch.pow(self.Q1(s, a) - y, 2).mean()
        q2_loss = torch.pow(self.Q2(s, a) - y, 2).mean()

        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        self.updates += 1
        if self.updates % 2 == 0:
            policy_loss = -self.Q1(s, self.μ(s)).mean()
            self.μ.optimize(policy_loss)

            self.μ.soft_update_target()
            self.Q1.soft_update_target()
            self.Q2.soft_update_target()
