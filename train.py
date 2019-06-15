from rl import *

# PROXIMAL POLICY OPTIMIZATION: ANT
class Config:
    env = 'Ant-v2'
    max_eps = 1000
    trajectory_length = 200
    vis_iter = 1
    storage_size = None
    batch_size = 10
    epochs = 4

agent = Agent(PPO, Config)
agent.train()

# SOFT ACTOR-CRITIC: PENDULUM
# class Config:
#     env = 'Pendulum-v0'
#     max_eps = 100
#     trajectory_length = 1
#     vis_iter = 1
#     storage_size = 1000000
#     batch_size = 128
#     epochs = 1
#
# agent = Agent(SAC, Config)
# agent.train()
