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

def load_context(path_to_weights, args, dims, device):
    """
    Load Pretrained context vector AutoEncoder

    These params will be trainable during the RL pretraining window.
    """
    optim_args = (args.lr, (0, 0.99))
    encoder_args = {'batch_size': args.batch, 'window_size': args.w, 'latent': dims[0]*dims[1],
                    'dims': dims}
    decoder_args = {}
    model = VGG16_AE(encoder_args, decoder_args, device)
    model, optimizer = model.load_model(path_to_weights, device, optim_args)
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
    initial_fund = args.initial
    trade_price = args.tp
    num_tickers = dataset.shape[1]
    max_hold = args.mh
    
    window = args.w
    iters = args.iters
    tol = args.tol

    ### - Create Agent - ###
    trader = Agent(alpha, beta, lr, dataset.shape, tau, '~/tmp/checkpoint', 's&ptrader')

    ### - env - ###
    env = PreTrainEnv(initial_fund, trade_price, num_tickers, max_hold, max_stock_value, 
                      window, iters, dataset, tol, trader)

    print('--------------- Begin Training ---------------')
    np.random.seed(0)
    score_history = []
    for i in range(int(iters)):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = trader.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            trader.remember(obs, act, reward, new_state, int(done))
            trader.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        if i % 100 == 0:
            trader.save_models()

    print('Done Training')