import gym
import torch

class Env:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.action_space.low = torch.FloatTensor(self.action_space.low)
        self.action_space.high = torch.FloatTensor(self.action_space.high)

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a.numpy())

class TanhAction(Env):
    def __init__(self, env_name):
        super().__init__(env_name)

    def step(self, a):
        a = a = ((a + 1) / 2) * (self.action_space.max - self.action_space.min) + self.action_space.min
        return super().step(a)
