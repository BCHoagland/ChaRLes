import torch
from rl.env import *
from rl.storage import *
from rl.visualize import *

class Agent:
    def __init__(self, algo, config):
        self.env = Env(config.env)
        try:
            for env_wrapper in self.algo.env_wrappers:
                self.env = env_wrapper(self.env)
        except:
            pass

        self.visualizer = Visualizer()
        self.config = config

        self.storage = Storage(config)

        self.algo = algo()
        self.algo.agent = self
        self.algo.setup()

    def explore(self):
        s = self.env.reset()
        for step in range(int(10000)):
            a = torch.FloatTensor(self.env.action_space.sample())
            s2, r, done, _ = self.env.step(a)
            self.storage.store((s, a, r, s2, done))
            s = self.env.reset() if done else s2

    def train(self):
        ep = 0
        ep_reward = 0
        s = self.env.reset()
        while ep < self.config.max_eps:
            t = 0
            while t < self.config.trajectory_length:
                with torch.no_grad():
                    s2, r, done, data = self.algo.interact(s)
                ep_reward += r
                self.storage.store(data)

                s = s2
                if done:
                    s = self.env.reset()
                    ep += 1
                    if ep % self.config.vis_iter == 0:
                        self.visualizer.update_viz(ep, ep_reward, self.config.env, self.algo.name, self.algo.color)
                    ep_reward = 0
                t += 1

            for _ in range(self.config.epochs):
                self.algo.update(self.storage)
            if self.algo.type == 'on-policy':
                self.storage.clear()
