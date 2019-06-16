from rl import *

class PGConfig:
    env = 'Ant-v2'
    max_eps = 1000
    trajectory_length = 200
    vis_iter = 1
    storage_size = None
    batch_size = 10
    epochs = 4

class QConfig:
    env = 'Ant-v2'
    max_eps = 200
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1000000
    batch_size = 128
    epochs = 1

setups = [
    # (PPO, PGConfig),
    # (DDPG, QConfig),
    # (TD3, QConfig),
    (SAC, QConfig),
]

for algo, config in setups:
    agent = Agent(algo, config)
    agent.train()
