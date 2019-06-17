from charles import *

class Config:
    env = 'Pendulum-v0'
    actors = 4
    max_timesteps = 3e4
    trajectory_length = 1
    vis_iter = 500
    storage_size = 1000000
    batch_size = 128
    epochs = 1

agent = Agent(DDPG, Config)
agent.train()

agent = Agent(TD3, Config)
agent.train()

# agent = Agent(SAC, Config)
# agent.train()

# class Config:
#     env = 'CartPole-v1'
#     actors = 4
#     max_timesteps = 4e4
#     trajectory_length = 1
#     vis_iter = 500
#     storage_size = 1000000
#     batch_size = 128
#     epochs = 1
#
# agent = Agent(DQN, Config)
# agent.train()
