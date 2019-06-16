from charles import *

class Config:
    env = 'Pendulum-v0'
    max_eps = 150
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1000000
    batch_size = 128
    epochs = 1

agent = Agent(SAC, Config)
agent.train()
