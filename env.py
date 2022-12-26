### - Standard NN Imports - ###
import gym
import torch as T
import torch.nn.functional as F
import numpy as np
import pandas as pd
from gym import spaces

### - internal libraries - ###
from forecasting_net.utils import *

### - visualization - ###
import matplotlib.pyplot as plt
import seaborn as sns

### - logging - ###
import tensorboard
from torch.utils.tensorboard import SummaryWriter

class PreTrainEnv(gym.env):
    def __init__(self, initial_fund, trade_price, dims, num_tickers, max_hold):
        super(PreTrainEnv, self).__init__()

        ### - Reset Vars - ###
        self.initial = initial_fund
        self.trade_price = trade_price
        self.data = T.zeros_like(dims)
        self.time_init = 0
        self.num_tickers = num_tickers

        ### - Attributes - ###
        self.dims = dims
        self.timestep = 0  
        self.net_worth = self.initial
        self.max_hold = max_hold

        ### - Spaces - ###
        self.action_space = self.gen_action_space() #TODO: gym.Spaces framework
        self.observation_space = self.gen_obs_space() #TODO: gym.Spaces framework
    
    def gen_action_space(self):
        """
        Agent has control over integer counts of each stock. At each timestep during forward testing amounts will be
        tracked and differences in amounts at time T will mean buy/sell/hold commands of that difference
        """
        return spaces.Box(low=np.zeros(self.num_tickers), high=self.max_hold*np.ones(self.num_tickers), dtype=np.int32)
    
    def gen_obs_space(self):
        raise NotImplementedError



