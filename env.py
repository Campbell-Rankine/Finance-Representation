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
    def __init__(self, initial_fund, trade_price, dims, num_tickers, max_hold, max_stock_value, window, _iters_):
        super(PreTrainEnv, self).__init__()

        ### - Reset Vars - ###
        self.initial = initial_fund
        self.trade_price = trade_price
        self.data = T.zeros_like(dims)
        self.time_init = 0
        self.num_tickers = num_tickers

        ### - Shape - ###
        self.dims = dims
        self.window = window

        ### - Environment Attributes - ###
        self.timestep = 0
        self.MAX_ITERS = _iters_
        self.available_funds = self.initial
        self.max_hold = max_hold
        self.max_price = max_stock_value

        ### - Spaces - ###
        self.action_space = self.gen_action_space() 
        self.observation_space = self.gen_obs_space() 

        ### - Timestep vars - ###
        self.holdings = np.zeros(num_tickers, dtype=np.int64)
        self.current_prices = np.zeros(num_tickers, dtype=np.float32)
    
    def gen_action_space(self):
        """
        Agent has control over integer counts of each stock. At each timestep during forward testing amounts will be
        tracked and differences in amounts at time T will mean buy/sell/hold commands of that difference
        """
        return spaces.Box(low=np.zeros(self.num_tickers), high=self.max_hold*np.ones(self.num_tickers), dtype=np.int32)
    
    def gen_obs_space(self):
        """
        Agent sees previous stock price across window of size self.window_size at each timestep t
        """
        return spaces.Box(low=np.zeros_like(self.dims), high=self.max_price*np.ones_like(self.dims), dtype=np.float32)
    
    def _get_observation(self):
        #TODO: Implement observation retrieval from dataset.
        raise NotImplementedError
    
    def _get_action(self):
        #TODO: Implement next action retrieval
        raise NotImplementedError
    
    def _get_current_prices(self):
        #TODO: Get current timestep stock prices
        raise NotImplementedError

    def _reward(self):
        curr_worth = self.holdings @ self.current_prices
        #TODO: Risk balancing, fund management, etc.
        time_pen = self.timestep / self.MAX_ITERS
        return curr_worth * time_pen

    def step(self):
        #update for reward
        self.holdings = self._get_action()
        self.current_prices = self._get_current_prices()
        self.timestep += 1

        #get reward and done flag
        worth = self._reward()
        done = worth <= 0 or self.available_funds < 0

        #next observation
        obs = self._get_observation()

        return obs, worth, done, {}





