import gym
from rl import *

class Config:
    max_eps = 1000
    trajectory_length = 'ep'
    vis_iter = 10
    storage_size = None

env = gym.make('CartPole-v1')
config = Config()

agent = Agent(env, PG, Config)
agent.train()


# class Config:
#     max_eps = 200
#     trajectory_length = 1
#     vis_iter = 1
#     storage_size = 1000000
#     batch_size = 128
#
# env = gym.make('Pendulum-v0')
# config = Config()
#
# agent = Agent(env, DDPG, Config)
# agent.train()
