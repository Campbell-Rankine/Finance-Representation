import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import gym
from tqdm import tqdm

import numpy as np
import pandas as pd

from Trader.trade_utils import *
from Trader.constants import *
from Trader.networks import *
from env import *
from forecasting_net.pred_network import *

def build_agent(args):
    raise NotImplementedError

def data_context_input_transform(x):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(x)

def load_context(path_to_weights, args, dims, device, pre=False):
    """
    Load Pretrained context vector AutoEncoder

    These params will be trainable during the RL pretraining window.
    """
    optim_args = (args.lr, (0, 0.99))
    encoder_args = {'batch_size': args.batch, 'window_size': args.w, 'latent': dims[0]*dims[1],
                    'dims': dims}
    decoder_args = {}
    model = VGG16_AE(encoder_args, decoder_args, device)
    optim_args = (lr, (0, 0.99))
    if pre:
        model, optimizer = model.load_model(path_to_weights, device, optim_args)
    else:
        optimizer = T.optim.Adam(model.parameters(), *optim_args)
    return model, optimizer

if __name__ == '__main__':
    """
    Main function
    """
    print('Begin Training')
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    args = process_command_line_arguments_()

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

    #names = ['decision_actor', 'decision_critic', 'target_actor', 'target_critic']
    
    ### - Get dataset as numpy array, convert using ray core API - ###
    dataset = load_dataset(dataset_path)
    print('Building Dataset')
    dataset = build_dataset(dataset)
    print('Built')

    ### - Load context model - ###
    print('load context model')
    model, optimizer = load_context(context_path, args, (6,331), device)
    print('Done')

    ### - env args - ###
    trade_price = args.tp
    print(dataset.shape)
    num_tickers = int(dataset.shape[0] / 6) - 1
    max_hold = args.mh
    
    window = args.w
    iters = args.iters
    tol = args.tol

    ### - Create Agent - ###
    trader = Agent(alpha, beta, lr, dataset.shape, tau, '/Users/bigc/Documents/Code - Offline/checkpoint', 's&ptrader')

    ### - env - ###
    env = PreTrainEnv(initial_fund, trade_price, num_tickers, max_hold, max_stock_value, 
                      window, iters, dataset, tol, trader)

    print('--------------- Begin Training ---------------')
    np.random.seed(0)
    score_history = []
    databar = tqdm(range(int(iters)))
    last_sample_mag = 0.
    for i in databar:
        if i == 0:
            print(env.initial)
        obs = env.reset()
        score = 0
        done = False
        _score_ = 0
        j = 1
        while not done:
            act = trader.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            if new_state.shape[0] < 30:
                print()
                break
            trader.remember(obs, act, reward, new_state, int(done))
            trader.learn()
            _score_ += (reward / j)
            obs = new_state
            databar.set_description('ep %i score %.5f, 100 games avg %.3f, score %.5f, current %.5f, init: %.5f, exp %.5f, avail: %.5f' % 
                (i, _score_, np.mean(score_history[-100:]), reward, env.worth, env.i_worth, last_sample_mag, env.available_funds))
            score_history.append(_score_)
            last_sample_mag = trader.sample_mag
            j = j + 1
        if i % 50 == 0 and env.tolerance >= 1:
            env.tolerance -= 1
        databar.set_description('ep %i score %.5f, 100 games avg %.3f, score %.2f \n current %.5f, init: %.5f, exp %.5f' % 
            (i, score, np.mean(score_history[-100:]), 0., env.worth, env.i_worth, last_sample_mag))
        if i % 1000 == 0:
            trader.save_models()

    print('Done Training')