import random
import torch
import numpy as np
from collections import deque

# def fix_d(d):
#     if isinstance(d, bool):
#         d = 1 - d
#     d = np.array(d)
#     # if len(d.shape) == 0:
#     #     d = np.expand_dims(d, axis=0)
#     return torch.FloatTensor(d)
#
# class Storage:
#     def __init__(self, size=None):
#         self.data = deque(maxlen=size)
#
#     def store(self, data):
#         self.data.append([fix_d(d) for d in data])
#
#     def get_all(self):
#         def fix(arr):
#             try:
#                 return torch.stack(arr)
#             except:
#                 return torch.FloatTensor(arr)
#         n = len(self.data[0])
#         data = [fix([d[i] for d in self.data]) for i in range(n)]
#         return data
#
#     def clear(self):
#         self.data.clear()

class Storage():
    def __init__(self, config):
        self.buffer = deque(maxlen=config.storage_size)
        self.config = config

    def store(self, data):
        def fix(x):
            if isinstance(x, bool): return 1 - x
            if not isinstance(x, np.ndarray): return np.array(x)
            else: return x

        transition = tuple(fix(x) for x in data)
        self.buffer.append(transition)

    def get(self, source):
        n = len(self.buffer[0])
        data = [torch.FloatTensor(np.array([arr[i] for arr in source])) for i in range(n)]
        max_dim = max([len(d.shape) for d in data])
        for i in range(len(data)):
            while len(data[i].shape) < max_dim:
                data[i].unsqueeze_(1)
        return data

    def get_all(self):
        return self.get(self.buffer)

    def sample(self):
        batch_size = min(len(self.buffer), self.config.batch_size)
        batch = random.sample(self.buffer, batch_size)
        return self.get(batch)

    def clear(self):
        self.buffer.clear()
