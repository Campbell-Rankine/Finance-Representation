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
    def __init__(self, initial_fund, trade_price, dims):
        super(PreTrainEnv, self).__init__()
        
        ### - Reset Vars - ###
        self.initial = initial_fund
        self.trade_price = trade_price
        self.data = T.zeros_like(dims)
        self.time_init = 0

        ### - Attributes - ###
        self.dims = dims
        self.timestep = 0  
        self.net_worth = self.initial

        ### - Spaces - ###
        self.action_space = None #TODO: gym.Spaces framework
        self.observation_space = None #TODO: gym.Spaces framework



