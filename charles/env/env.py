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

    def explore_step(self, a):
        s2, r, done, info = self.env.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info

    def step(self, a):
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        s2, r, done, info = self.env.step(a)
        if len(np.array(s2).shape) == 0:
            s2 = np.expand_dims(s2, axis=0)
        return s2, r, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

class TanhAction(Env):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, k):
        return getattr(self.env, k)

    def step(self, a):
        a = ((torch.FloatTensor(a) + 1) / 2) * (self.env.action_space.high - self.env.action_space.low) + self.env.action_space.low
        return self.env.step(a)
