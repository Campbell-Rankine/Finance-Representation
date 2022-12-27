import torch as T
import torch.nn as nn
import torch.nn.functional as F

from trade_utils import *

import os
import ray

### - DEFINE ACTOR CRITIC NETWORKS - ###

class Actor(nn.Module):
    def __init__(self, h1, h2, obs_size, action_size, batch_size, tau, betas, lr, cp, name, device):
        super(Actor, self).__init__()
        
        ### - Attributes - ###
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.cp_save = os.path.join(cp, name)

        ### - Network Modules - ###
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.obs_size, h1)
        self.f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1, -self.f1, self.f1)

        self.ln1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1, h2)
        self.f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2, -self.f2, self.f2)

        self.ln2 = nn.LayerNorm(h2)

        self.mu = nn.Linear(h2, self.action_size)
        self.f3 = 0.003
        T.nn.init.uniform_(self.mu, -self.f3, self.f3)

        self.optim = T.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.device = device

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        x = T.tanh(self.mu(x))
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.cp_save)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.cp_save))

class Critic(nn.Module):
    def __init__(self, h1, h2, obs_size, action_size, batch_size, w_decay, tau, betas, lr, cp, name, device):
        super(Actor, self).__init__()
        
        ### - Attributes - ###
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.cp_save = os.path.join(cp, name)
        self.w_decay = w_decay

        ### - Network Modules - ###
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(self.obs_size, h1)
        self.f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1, -self.f1, self.f1)

        self.ln1 = nn.LayerNorm(h1)

        self.fc2 = nn.Linear(h1, h2)
        self.f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2, -self.f2, self.f2)

        self.ln2 = nn.LayerNorm(h2)

        self.q = nn.Linear(h2, 1)
        self.f3 = 0.003
        T.nn.init.uniform_(self.q, -self.q, self.q)

        self.av = nn.Linear(h2, self.action_size)

        self.optim = T.optim.Adam(self.parameters(), lr=lr, betas=betas, weight_decay=self.w_decay)
        self.device = device

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)

        action_value = self.activation(self.av(action))
        state_action_value = self.activation(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.cp_save)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.cp_save))

### - DEFINE AGENT - ###