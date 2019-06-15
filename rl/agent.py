import gym
import torch
from rl.storage import *
from rl.visualize import *

class Agent:
    def __init__(self, algo, config):
        self.env = gym.make(config.env)
        self.config = config

        self.storage = Storage(config)

        self.algo = algo()
        self.algo.agent = self
        self.algo.setup()

    def explore(self):
        s = self.env.reset()
        for step in range(int(10000)):
            a = self.env.action_space.sample()
            s2, r, done, _ = self.env.step(a)
            self.storage.store((s, a, r, s2, done))
            s = self.env.reset() if done else s2

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
