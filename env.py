### - Standard NN Imports - ###
import gym
from gym import Env
import torch as T
import torch.nn.functional as F
import numpy as np
import pandas as pd
from gym import spaces
import random

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

#TODO: Weird reset bug? Goes to the millions for some reason, should probably fix that

class PreTrainEnv(Env):
    def __init__(self, initial_fund, trade_price, num_tickers, max_hold, max_stock_value, window, _iters_, dataset,
                tolerance, agent):
        super(PreTrainEnv, self).__init__()

        ### - Reset Vars - ###
        self.i_worth = None
        self.agent = agent
        self.initial = initial_fund
        self.trade_price = trade_price
        self.data = dataset #(DATASET FORMAT, self.dims = (num_tickers, num_features, num_samples))

        self.time_init = window + 1
        self.num_tickers = num_tickers

        ### - Shape - ###
        self.window = window
        self.dims = (self.agent.obs_size, self.window)

        ### - Environment Attributes - ###
        self.timestep = 0
        self.MAX_ITERS = _iters_
        self.available_funds = 70000
        self.max_hold = max_hold
        self.max_price = max_stock_value
        self.tolerance = tolerance

        self.last_10 = []

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
        obs = self.data[max(0, (self.timestep - self.window)):self.timestep]
        return obs
    
    def _get_current_prices(self, obs):
        """
        For the sake of adding noise sample a random price from somewhere within the [t - tolerance, t + tolerance] range

        """
        indices = list(range(6, self.data.shape[1], 6))
        t_ind = np.random.randint(0, len(obs))
        pricing = np.array([x[indices] for x in obs])
        return pricing[t_ind]

    def _reward(self):
        self.worth = self.holdings @ self.current_prices
        self.last_10.append(self.worth)
        #self.agent.holdings = self.worth
        self.i_worth = self.initial @ self.initial_p
        #self.net_change = self.worth - self.i_worth
        #TODO: Risk balancing, fund management, etc.
        time_pen = self.timestep / len(self.data)
        rew_ = ((self.worth * time_pen) / max(self.last_10)) + ((self.worth - self.last_10[-1]) / max(self.last_10))
        return rew_

    def update_inventory(self):
        """
        Write function to change amount to purchase, basically conducting the buy sell commands.
        """
        raise NotImplementedError

    def step(self, action):
        #update for reward
        self.holdings = action
        self.timestep += 1

        if len(self.last_10) > 10:
            self.last_10 = self.last_10[2:]

        #get reward and done flag
        worth = self._reward()
        done = False
        self.prev_reward = self._reward()
        if self.worth <= 0:
            done = True
        #print(self.available_funds, self.timestep, len(self.data))
        if self.available_funds < 0 or self.timestep > len(self.data):
            done = True

        #next observation
        obs = self._get_observation()
        risk = np.std(obs)
        risk = risk / max(np.std(obs, axis=1))
        #print(risk, worth)
        worth = worth - risk #Risk penalty if we follow R_set ~ Var[Observation]
        self.current_prices = self._get_current_prices(obs)

        return obs, worth, done, {}


    def reset(self):
        ### - Resets - ###
        self.i_worth = None
        self.holdings = np.zeros(self.num_tickers)
        self.available_funds = 70000
        self.timestep = self.time_init #TODO: add random timestep init

        ### - first obs - ###
        obs = self._get_observation()
        self.current_prices = self._get_current_prices(obs)
        self.initial_p = self._get_current_prices(obs)
        #print(self.current_prices)
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


"""
SUGGESTIONS:
    1) Probably need a better rendering function, something that more accurately, and visually displays the model
    decision making process. Maybe take the train functions from GANS that produce GIFS?

    2) Need to play around with reward functions that factor in some kind of 'risk' dampener, It's up to you to
    research how risk is defined in your setting, but lets maybe make that some kind of control setting in the
    final deployment

    3) would be nice to output Gradients and network penalties to visualize learning so have a look for how you can do that

    4) CPU paralellize the RUN and TRAIN functions when theyre working
"""