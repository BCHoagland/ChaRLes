import torch
from rl.storage import *
from rl.visualize import *

class Agent:
    def __init__(self, env, algo, config):
        self.env = env
        self.config = config

        self.algo = algo()
        self.algo.setup(env)

        self.storage = Storage()

    def train(self):
        ep = 0
        ep_reward = 0
        s = self.env.reset()
        while ep < self.config.max_eps:
            t = 0
            while self.config.trajectory_length == 'ep' or t < self.config.trajectory_length:
                with torch.no_grad():
                    s2, r, done, data = self.algo.interact(s)
                ep_reward += r
                self.storage.store(data)

                s = s2
                if done:
                    s = self.env.reset()
                    ep += 1
                    if ep % self.config.vis_iter == self.config.vis_iter - 1:
                        update_viz(ep, ep_reward, self.algo.name)
                    ep_reward = 0

                    if self.config.trajectory_length == 'ep':
                        self.algo.update(self.storage)
                        break
                t += 1

            if self.config.trajectory_length != 'ep':
                self.algo.update(self.storage)