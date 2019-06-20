from charles import *

class Config:
    env = 'Pendulum-v0'
    actors = 8
    lr = 3e-4
    max_timesteps = 1e4
    trajectory_length = 1
    vis_iter = 500
    storage_size = 1000000
    batch_size = 128
    epochs = 1
    explore_steps = 10000

agent = Agent(SAC, Config)
agent.train()
