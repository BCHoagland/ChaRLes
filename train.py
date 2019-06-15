from rl import *

# POLICY GRADIENT: CARTPOLE
class Config:
    env = 'CartPole-v1'
    max_eps = 300
    trajectory_length = 'ep'
    vis_iter = 10
    storage_size = None

agent = Agent(PG, Config)
agent.train()

# SOFT ACTOR-CRITIC: PENDULUM
class Config:
    env = 'Pendulum-v0'
    max_eps = 100
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1000000
    batch_size = 128

agent = Agent(SAC, Config)
agent.train()
