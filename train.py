import charles

# class Config:
#     env = 'Pendulum-v0'
#     actors = 4
#     lr = 3e-4
#     max_timesteps = 1e4
#     trajectory_length = 1
#     vis_iter = 200
#     storage_size = 1000000
#     batch_size = 512
#     epochs = 1
#     explore_steps = 1000

# agent = charles.Agent(charles.algos.SAC, Config)
# agent.train()

class Config:
    env = 'CartPole-v1'
    actors = 4
    lr = 3e-4
    max_timesteps = 1e5
    trajectory_length = 1
    vis_iter = 500
    storage_size = 1000000
    batch_size = 256
    epochs = 1
    explore_steps = 10000

# agent = charles.Agent(charles.algos.SAC_d, Config)
agent = charles.Agent(charles.algos.DQN, Config)
agent.train()