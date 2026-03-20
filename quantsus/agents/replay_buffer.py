import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size=500):
        self.ptr = 0
        self.size = 0
        self.max_size = size

        self.state = np.zeros((size, state_dim))
        self.action = np.zeros((size, action_dim))
        self.reward = np.zeros((size, 1))
        self.next_state = np.zeros((size, state_dim))
        self.done = np.zeros((size, 1))

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        idx = np.random.randint(0, self.size, size=batch_size)

        return dict(
            state=torch.FloatTensor(self.state[idx]),
            action=torch.FloatTensor(self.action[idx]),
            reward=torch.FloatTensor(self.reward[idx]),
            next_state=torch.FloatTensor(self.next_state[idx]),
            done=torch.FloatTensor(self.done[idx]),
        )