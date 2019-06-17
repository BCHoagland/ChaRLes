import random
from charles.algorithm import Algorithm
from charles.models import *
from charles.utils import *

class DDQN(Algorithm):
    def setup(self):
        self.name = 'DDQN'
        self.type = 'off-policy'
        self.color = [248, 136, 112]

        self.Q1 = Model(DQNNet, self.env, 1e-3, target=True)
        self.Q2 = Model(DQNNet, self.env, 1e-3, target=True)

        self.explore()

    def interact(self, s):
        if random.random() < 0.05:
            a = np.stack([self.env.action_space.sample() for _ in range(self.config.actors)])
        else:
            a = np.argmax(self.Q1(s), axis=1).numpy()

        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, (data)

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        a2 = torch.argmax(self.Q1.target(s2), dim=2, keepdim=True)
        next_q = self.Q2.target(s2).gather(2, a2.long())
        y = r + (0.99 * m * next_q)

        q1_loss = torch.pow(self.Q1(s).gather(2, a.long()) - y, 2).mean()
        q2_loss = torch.pow(self.Q2(s).gather(2, a.long()) - y, 2).mean()

        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        self.Q1.soft_update_target()
        self.Q2.soft_update_target()