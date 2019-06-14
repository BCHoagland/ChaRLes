import gym
from rl import *

class Config:
    max_eps = 1000
    trajectory_length = 'ep'
    vis_iter = 10

env = gym.make('CartPole-v1')
config = Config()

agent = Agent(env, PG, Config)
agent.train()
