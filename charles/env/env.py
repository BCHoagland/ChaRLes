import gym
import torch
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_env(id):
    def _f():
        return gym.make(id)
    return _f

class Env:
    def __init__(self, env_name, actors):
        # self.env = gym.make(env_name)
        self.env = SubprocVecEnv([make_env(env_name) for _ in range(actors)])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        try:
            self.action_space.low = torch.FloatTensor(self.action_space.low)
            self.action_space.high = torch.FloatTensor(self.action_space.high)
        except:
            self.action_space.low = None
            self.action_space.high = None

    def reset(self):
        s = self.env.reset()
        if len(np.array(s).shape) == 0:
            s = np.expand_dims(s, axis=0)
        return s

    def step(self, a):
        # if len(np.array(a).shape) == 0:
        #     a = torch.FloatTensor([a])
        # else:
        #     a = torch.FloatTensor(a)

        # if self.env.action_space.__class__.__name__ == 'Discrete':
        #     a = int(a.item())
        # else:
        #     a = a.numpy()

        s2, r, done, info = self.env.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

class TanhAction(Env):
    def __init__(self, env_name):
        super().__init__(env_name)

    def step(self, a):
        a = a = ((a + 1) / 2) * (self.action_space.max - self.action_space.min) + self.action_space.min
        return super().step(a)
