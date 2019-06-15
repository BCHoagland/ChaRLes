from rl.algorithm import Algorithm
from rl.models import *
from rl.utils import *

class SAC(Algorithm):
    def setup(self):
        self.name = 'SAC'
        self.type = 'off-policy'

        self.π = Model(TanhPolicy(self.env), 3e-4)
        self.Q1 = Model(Q(self.env), 3e-4, target=True)
        self.Q2 = Model(Q(self.env), 3e-4, target=True)
        self.α = LearnableParam(0.2, 1e-4)
        self.target_entropy = -torch.prod(torch.FloatTensor(self.env.action_space.shape)).item()

        self.explore()

    def interact(self, s):
        a = self.π(s)
        s2, r, done, _ = self.env.step(a * 2)               # FIX WITH ENV WRAPPER
        data = (s, a, r, s2, done)
        return s2, r, done, data

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        with torch.no_grad():
            a2, p2, = self.π.sample(s2)
            min_next_q = torch.min(self.Q1.target(s2, a2), self.Q2.target(s2, a2))
            y = r + (0.99 * m * min_next_q)

        q1_loss = torch.pow(self.Q1(s, a) - y, 2).mean()
        q2_loss = torch.pow(self.Q2(s, a) - y, 2).mean()
        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        new_a, p = self.π.sample(s)
        min_q = torch.min(self.Q1(s, new_a), self.Q2(s, new_a))
        policy_loss = (self.α * p - min_q).mean()
        self.π.optimize(policy_loss)

        α_loss = -(self.α * (p + self.target_entropy).detach()).mean()
        self.α.optimize(α_loss)

        self.Q1.soft_update_target()
        self.Q2.soft_update_target()
