import numpy as np
dataset_path = '/Users/bigc/Documents/Code - Offline/dataset_long_ohlc.pickle'
context_path = '/Users/bigc/Documents/Code - Offline/SM_Representation_Net.pth'
debug_len = 10
max_stock_value = 100000000
initial_fund = np.random.randint(low=0, high=5, size=138)