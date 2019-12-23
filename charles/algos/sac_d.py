from charles.algorithm import Algorithm
from charles.models import *
from charles.env import *

class SAC_d(Algorithm):
    def __init__(self):
        self.name = 'SAC Discrete'
        self.type = 'off-policy'
        self.color = [0, 0, 0]
        # self.env_wrappers = [TanhAction]
    

    def setup(self):
        self.π = Model(d_Policy, self.env, self.config.lr)
        self.Q1 = Model(d_Q, self.env, self.config.lr, target=True)
        self.Q2 = Model(d_Q, self.env, self.config.lr, target=True)

        self.α = LearnableParam(0.2, self.config.lr)
        # self.target_entropy = -torch.prod(torch.FloatTensor(self.env.action_space.shape)).item()

        # self.explore()
    

    def interact(self, s):
        a = self.π.select_action(s)
        s2, r, done, _ = self.env.step(a)
        data = (s, a, r, s2, done)
        return s2, r, done, data
    

    def update(self, storage):
        s, a, r, s2, m = storage.sample()

        #! print(s.shape, a.shape, r.shape, s2.shape, m.shape)
        batch_dim = s.shape[0]

        # optimize Q networks
        with torch.no_grad():
            next_q = self.Q1.target(s2)                                             #? Bellman backups calculated with Q1
            diff = next_q - (self.α * self.π.log_probs(s2))
            a2 = self.π(s2)
            expected_q = torch.bmm(
                diff.view(batch_dim, 1, 2),
                a2.view(batch_dim, 2, 1)
            ).squeeze().unsqueeze(1)
            y = r + (0.99 * m * expected_q)
        
        q1_loss = ((self.Q1(s).gather(-1, a.long()) - y) ** 2).mean()
        q2_loss = ((self.Q2(s).gather(-1, a.long()) - y) ** 2).mean()
        self.Q1.optimize(q1_loss)
        self.Q2.optimize(q2_loss)

        # optimize policy
        #! min_q = torch.min(self.Q1(s), self.Q2(s))
        diff = self.α * self.π.log_probs(s) - self.Q2(s)                            #? KL Divergence calculated with Q2
        p = self.π(s)
        expected_pi = torch.bmm(
            diff.view(batch_dim, 1, 2),
            p.view(batch_dim, 2, 1)
        ).squeeze().mean()
        policy_loss = expected_pi.mean()
        self.π.optimize(policy_loss)

        # α_loss = -(self.α * (p + self.target_entropy).detach()).mean()
        # self.α.optimize(α_loss)

        self.Q1.soft_update_target()
        # self.Q2.soft_update_target()