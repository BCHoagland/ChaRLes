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
    def __init__(self, size=None):
        self.buffer = deque(maxlen=size)

    def get_all(self):
        s_arr = torch.FloatTensor(np.array([arr[0] for arr in self.buffer]))
        a_arr = torch.FloatTensor(np.array([arr[1] for arr in self.buffer]))
        r_arr = torch.FloatTensor(np.array([arr[2] for arr in self.buffer]))
        m_arr = torch.FloatTensor(np.array([arr[3] for arr in self.buffer]))

        return s_arr, a_arr, r_arr.unsqueeze(1), m_arr.unsqueeze(1)

    def store(self, data):
        def fix(x):
            if isinstance(x, bool): return 1 - x
            if not isinstance(x, np.ndarray): return np.array(x)
            else: return x

        transition = tuple(fix(x) for x in data)
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()
