import random
import torch
import numpy as np
from collections import deque

class Storage():
    def __init__(self, config):
        self.buffer = deque(maxlen=config.storage_size)
        self.config = config

    def store(self, data):
        '''stored a single group of data'''
        def fix(x):
            if isinstance(x, bool): return 1 - x
            if not isinstance(x, np.ndarray): return np.array(x)
            else: return x

        transition = tuple(fix(x) for x in data)
        self.buffer.append(transition)

    def get(self, source):
        '''return all data from the given source'''

        # group together all data of the same type
        n = len(self.buffer[0])
        data = [torch.FloatTensor(np.array([arr[i] for arr in source])) for i in range(n)]

        # expend data dimensions until they all have the same number of dimensions
        max_dim = max([len(d.shape) for d in data])
        for i in range(len(data)):
            while len(data[i].shape) < max_dim:
                data[i].unsqueeze_(1)
        return data

    def get_all(self):
        '''return all stored data'''
        return self.get(self.buffer)

    def sample(self):
        '''return a random sample from the stored data'''
        batch_size = min(len(self.buffer), self.config.batch_size)
        batch = random.sample(self.buffer, batch_size)
        return self.get(batch)

    def get_batches(self):
        # N = len(self.buffer)
        # batch_size = min(N, self.config.batch_size)
        # num_batches = len(self.buffer) // batch_size
        #
        # i = list(range(1, N))
        # random.shuffle(i)
        #
        # for batch in range(num_batches):
        #     yield self.get(self.buffer[])

        # TODO: shuffle all data, not just sample randomly multiple times

        num_batches = max(1, len(self.buffer) // self.config.batch_size)
        for _ in range(num_batches):
            yield self.sample()

    def clear(self):
        '''clear stored data'''
        self.buffer.clear()
