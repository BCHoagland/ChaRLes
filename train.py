import charles

class Config:
    env = 'Pendulum-v0'
    actors = 4
    lr = 3e-4
    max_timesteps = 1e4
    trajectory_length = 1
    vis_iter = 200
    storage_size = 1000000
    batch_size = 512
    epochs = 1
    explore_steps = 1000

agent = charles.Agent(charles.algos.SAC, Config)
agent.train()