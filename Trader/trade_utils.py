import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import ray
from tqdm import tqdm
import sklearn

@ray.remote(max_retries=5)
def process(key, raw_data: dict):
    item = raw_data[key]
    #try:
    item = item.drop('ticker', axis=1)
    #except :
    return item

def build_dataset(raw_data: dict):
    keys = list(raw_data.keys())
    ids = [process.remote(key, raw_data) for key in keys]
    data_ = ray.get(ids)
    data = np.array(data_[0])
    for x in data_[1:]:
        if x.shape[0] == data.shape[0]:
            data = np.hstack((data, x))
    print(data.shape)
    return data.T
    
def load_dataset(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

class OrnsteinUhlenbeckActionNoise:
    """
    Technique used to sample temporally located actions on a continuous action space. This will function the same way
    as adding normally distributed noise to the policy gradients during discrete DDPG. I.E. This class
    controls action exploration
    """
    def __init__(self, action_dim, mu=0, theta = 0.15, sigma = 0.1):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.X = self.mu*np.zeros(self.action_dim)

    def reset(self):
        self.X = self.mu*np.zeros(self.action_dim)

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
import argparse
def process_command_line_arguments_() -> argparse.Namespace:
    """Parse the command line arguments and return an object with attributes
    containing the parsed arguments or their default values.
    """
    import json

    parser = argparse.ArgumentParser()

    ### - Global Params - ###
    parser.add_argument("-optim", "--optim", dest="optim", metavar="optim", default = False,
                        type=bool, help="self optimize using Bayesian Optimization during training")
    
    parser.add_argument("-d", "--d", dest="d", metavar="d", default = False,
                        type=bool, help="Download default s&p 500 data")
    
    parser.add_argument("-clip", "--clip", dest="clip", metavar="clip", default = False,
                        type=bool, help="Gradient clipping: clipped at 10")

    parser.add_argument("-initial", "--initial", dest="initial", metavar="initial", default = 1e5,
                        type=float, help="initial fund value")

    parser.add_argument("-tp", "--tp", dest="tp", metavar="tp", default = 100.,
                        type=float, help="trade price")

    parser.add_argument("-mh", "--mh", dest="mh", metavar="mh", default = 1e9,
                        type=float, help="Max holdings")

    parser.add_argument("-w", "--w", dest="w", metavar="w", default = 30,
                        type=int, help="Network history observation window")

    parser.add_argument("-iters", "--iters", dest="iters", metavar="iters", default = 100,
                        type=int, help="Max iterations pre epoch")

    parser.add_argument("-tol", "--tol", dest="tol", metavar="tol", default = 2,
                        type=int, help="Pricing tolerance for generating random pricing")

    parser.add_argument("-cp", "--cp", dest="cp", metavar="cp", default = '/tmp/cp',
                        type=str, help="store network checkpoints")
    
    ### - Network Params - ###
    parser.add_argument("-h1", "--h1", dest="h1", metavar="h1", default = 300,
                        type=int, help="1st Hidden Layer Size")
    
    parser.add_argument("-h2", "--h2", dest="h2", metavar="h2", default = 400,
                        type=int, help="2nd Hidden Layer Size")

    parser.add_argument("-batch", "--batch", dest="batch", metavar="batch", default = 64,
                        type=int, help="default batch size")

    parser.add_argument("-lr", "--lr", dest="lr", metavar="lr", default = 0.1,
                        type=float, help="default learning rate")
    
    parser.add_argument("-a", "--a", dest="a", metavar="a", default = (0.000025, 0.99),
                        type=tuple, help="Network alphas")

    parser.add_argument("-b", "--b", dest="b", metavar="b", default = (0.00025, 0.99),
                        type=tuple, help="Network Betas")

    parser.add_argument("-g", "--g", dest="g", metavar="g", default = 0.99,
                        type=float, help="Network Gamma")

    parser.add_argument("-t", "--t", dest="t", metavar="t", default = 0.1,
                        type=float, help="Network Tau")

    parser.add_argument("-wd", "--wd", dest="wd", metavar="wd", default = 0.1,
                        type=float, help="Network Weight Decay")

    parser.add_argument("-e", "--e", dest="e", metavar="epochs", default = 64,
                        type=int, help="default num epochs")

    parser.add_argument("-id", "--id", dest="in_dest", metavar="in_dest", default = None,
                        type=str, help="input destination. Either change input flag or change the destination in the constants.py folder")
    
    parser.add_argument("-adam", "--adam", dest="adam", metavar="adam", default = None,
                        type=tuple, help="Enter custom adam parameters")

    args = parser.parse_args()

    return args

from collections import deque
class plot_mem(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.storage = deque()
        self.num_items = 0

    def add(self, item):
        if self.num_items >= self.max_len:
            self.storage.popleft()
            self.storage.append(item)
        else:
            self.storage.append(item)
    def reset(self):
        self.storage = deque()
        self.num_items = 0
