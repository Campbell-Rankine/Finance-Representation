### - Standard NN Imports - ###
import gym
from gym import Env
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
from collections import deque
from Trader.trade_utils import plot_mem

class PreTrainEnv(Env):
    def __init__(self, initial_fund, trade_price, num_tickers, max_hold, max_stock_value, window, _iters_, dataset,
                tolerance, agent):
        super(PreTrainEnv, self).__init__()

        ### - Reset Vars - ###
        self.agent = agent
        self.initial = initial_fund
        self.trade_price = trade_price
        self.data = dataset #(DATASET FORMAT, self.dims = (num_tickers, num_features, num_samples))

        self.time_init = 1
        self.num_tickers = num_tickers

        ### - Shape - ###
        self.window = window
        self.dims = (self.agent.obs_size, self.window)

        ### - Environment Attributes - ###
        self.timestep = 0
        self.MAX_ITERS = _iters_
        self.available_funds = self.initial
        self.max_hold = max_hold
        self.max_price = max_stock_value
        self.tolerance = tolerance

        ### - Spaces - ###
        self.action_space = self.gen_action_space() 
        self.observation_space = self.gen_obs_space() 

        ### - Timestep vars - ###
        self.holdings = np.zeros(num_tickers, dtype=np.int64)
        self.current_prices = np.zeros(num_tickers, dtype=np.float32)

        ### - Rendering - ###
        self.storage = plot_mem(100)
    
    def gen_action_space(self):
        """
        Agent has control over integer counts of each stock. At each timestep during forward testing amounts will be
        tracked and differences in amounts at time T will mean buy/sell/hold commands of that difference
        """
        return spaces.Box(low=np.zeros(self.num_tickers), high=np.ones(self.num_tickers), dtype=np.int32)
    
    def gen_obs_space(self):
        """
        Agent sees previous stock price across window of size self.window_size at each timestep t
        """
        return spaces.Box(low=np.zeros_like(self.dims), high=self.max_price*np.ones_like(self.dims), dtype=np.float32)
    
    def _get_observation(self):
        obs = self.data[:][-max(0, (self.timestep - self.window)):-self.timestep]
        print(obs)
        return obs
    
    def _get_current_prices(self):
        """
        For the sake of adding noise sample a random price from somewhere within the [t - tolerance, t + tolerance] range
        """
        obs = self._get_observation()
        pricing = obs['close']
        return np.random.sample(pricing[self.tol:-self.tol])

    def _reward(self):
        curr_worth = self.holdings @ self.current_prices
        self.net_change = curr_worth - self.initial
        #TODO: Risk balancing, fund management, etc.
        time_pen = self.timestep / self.MAX_ITERS
        return self.net_change * time_pen

    def step(self, action):
        #update for reward
        self.holdings = action
        self.current_prices = self._get_current_prices()
        self.timestep += 1

        #get reward and done flag
        worth = self._reward()
        self.prev_reward = self._reward()
        done = worth <= 0 or self.available_funds < 0 or self.timestep > len(self.data)

        #next observation
        obs = self._get_observation()

        return obs, worth, done, {}


    def reset(self):
        ### - Resets - ###
        self.holdings = np.zeros(self.num_tickers)
        self.current_prices = np.zeros(self.num_tickers)
        self.available_funds = self.initial
        self.timestep = self.time_init #TODO: add random timestep init

        ### - first obs - ###
        obs = self._get_observation()

        #self.agent.reset()
        return obs

    def render(self, mode='human', close=False):
        print('Current holdings at %i: ' % self.timestep)
        print(self.holdings)
        print()
        print('Reward: %.2f' % self.prev_reward)
        print('Available Funds: %.2f' % self.available_funds)
        print('-----------------------------\n')

        self.storage.add(self.net_change)
        t_x = sns.scatterplot(list(range(self.timestep)), list(self.storage.storage), cmap=sns.color_palette("magma", as_cmap=True))
        plt.savefig(fname='Tmp/change_at_' + str(self.timestep) + '.png')
