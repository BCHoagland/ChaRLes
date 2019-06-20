import torch
from charles.visualize import *
from charles.agents.agent import Agent

class Imitator(Agent):
    def __init__(self, algo, config, expert):
        super().__init__(algo, config)

        self.config = config

        # create expert dataset
        s = self.env.reset()
        k = int(config.storage_size)
        for t in range(k):
            if t % (k / 20) == (k / 20) - 1:
                progress(t, k, 'Making expert dataset')
            with torch.no_grad():
                s, _, _, data = expert.algo.interact(s)
                self.storage.store(data)

    def train(self):
        self.algo.setup()

        # pre-train imitation agent using expert dataset
        k = int(self.config.pre_training_epochs)
        for t in range(k):
            losses = self.algo.update(self.storage)
            if t % (k / 20) == (k / 20) - 1:
                progress(t, k, 'Pre-training')
                self.visualizer.plot(self.algo.name, 'Pretraining Loss', 'Timesteps', t, losses, color=self.algo.color, title='Loss')

        # continue to train imitation agent on its own
        super().train()
