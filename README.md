# ChaRLes

ChaRLes is my personal library for implementing Deep RL algorithms and running experiments. Compatible with OpenAI Gym environments.

## Supported Algorithms

#### On-Policy
- [x] Policy Gradient (PG)
- [x] Proximal Policy Optimization (PPO)

#### Off-Policy
- [x] Deep Q Learning (DQN)
- [x] Double DQN (DDQN)
- [x] Deep Deterministic Policy Gradient (DDPG)
- [x] Twin Dueling DDPG (TD3)
- [x] Soft Actor-Critic (SAC) ***(continuous actions only)***




## Usage
Visdom must be running prior to training.

From the top level of the ChaRLes directory, create a file and include `from charles import *` at the top. This imports the various algorithm classes and allows you to construct an agent.

To train an agent, create a config class and give it the necessary environment information and hyperparameters as fields. Pass the class of the algorithm you want and the config class you just made into a new Agent, then invoke `agent.train()`

##### Example: train.py
```python
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

agent = Agent(SAC, Config)
agent.train()
```
