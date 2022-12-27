import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn

class OrnsteinUhlenbeckActionNoise:
    """
    Technique used to sample temporally located actions on a continuous action space. This will function the same way
    as adding normally distributed noise to the policy gradients during discrete DDPG. I.E. This class
    controls action exploration
    """
    def __init__(self, action_dim, mu=0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.X = self.mu*np.ones(self.action_dim)

    def reset(self):
        self.X = self.mu*np.ones(self.action_dim)

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

class MBSTD(nn.Module):
    def __init__(self):
        super(MBSTD, self).__init__()
    def forward(self, x):
        size = list(x.size())
        size[1] = 1
		
        std = T.std(x, dim=0)
        mean = T.mean(std)
        return T.cat((x, mean.repeat(size)),dim=1)


