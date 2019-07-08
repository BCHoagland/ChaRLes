from charles import *
import real_service

class Config:
    env = 'RealService-v0'
    actors = 1
    lr = 3e-4
    max_timesteps = 5e4
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1e6
    batch_size = 128
    epochs = 1
    explore_steps = 0

agent = Agent(SAC, Config)
agent.train()
