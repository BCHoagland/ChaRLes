# ChaRLes

ChaRLes is my personal library for implementing Deep RL algorithms and running experiments

### Supported Algorithms

##### On-Policy
- [x] Policy Gradient (PG)
- [x] Proximal Policy Optimization

##### Off-Policy
- [x] Deep Deterministic Policy Gradient (DDPG)
- [x] Twin Dueling DDPG (TD3)
- [x] Soft Actor-Critic (SAC) ***(continuous actions only)***




### Usage
From the top level of the ChaRLes directory, create a file and include `from rl import *` at the top.
To run an algorithm in an OpenAI Gym environment, create a class and give it the necessary environment information and hyperparameters as fields.
Pass the class of the algorithm you want and the config class you just made into a new Agent, then call `agent.train()`

##### Example: train.py
```python
class Config:
    env = 'Pendulum-v0'
    max_eps = 100
    trajectory_length = 1
    vis_iter = 1
    storage_size = 1000000
    batch_size = 128
    epochs = 1

agent = Agent(SAC, Config)
agent.train()
```
