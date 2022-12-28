import torch as T
import torch.nn as nn
import torch.nn.functional as F

import gym

import numpy as np
import pandas as pd

from Trader.trade_utils import *
from Trader.constants import *
from env import *

def learn():
    """
    Main function
    """
    print('Begin Training')
    args = process_command_line_arguments()
    ### - network args - ###
    lr = args.lr
    alpha = args.a
    beta = args.b
    gamma = args.g
    tau = args.t
    weight_decay = args.wd
    epochs = args.e
    h1 = args.h1
    h2 = args.h2
    cp = args.cp

    names = ['decision_actor', 'decision_critic', 'target_actor', 'target_critic']

    ### - Get dataset as numpy array - ###
    #TODO: Implement this part
    dataset = None

    ### - Create Agent - ###
    #TODO: Implement

    ### - env args - ###
    initial_fund = args.initial
    trade_price = args.tp
    dims=dataset[0].shape
    num_tickers = len(dataset)
    max_hold = args.mh
    max_stock_value = 10*np.max(dataset)
    window = args.w
    iters = args.iters
    tol = args.tol

    ### - env - ###
    env = PreTrainEnv()#TODO: args, do correctly
    env.reset()


if __name__ == 'main':
    learn()

