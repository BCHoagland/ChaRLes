import gym
from rl import *

# class Config:
#     env = 'CartPole-v1'
#     max_eps = 1000
#     trajectory_length = 'ep'
#     vis_iter = 10
#     storage_size = None
#
# agent = Agent(PG, Config)
# agent.train()


# class Config:
#     env = 'Pendulum-v0'
#     max_eps = 200
#     trajectory_length = 1
#     vis_iter = 1
#     storage_size = 1000000
#     batch_size = 128
#
# agent = Agent(TD3, Config)
# agent.train()

class Config:
    env = 'Pendulum-v0'
    max_eps = 200
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1000000
    batch_size = 128

agent = Agent(SAC, Config)
agent.train()
