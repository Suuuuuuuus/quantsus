import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .variable import *

LOG_STD_MIN = -20
LOG_STD_MAX = 2
HIDDEN_DIM = 256

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z)

        # log prob correction for Tanh
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)