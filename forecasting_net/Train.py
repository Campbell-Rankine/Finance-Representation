### - Torch imports - ###
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

### - utils - ###
from utils import *
import pickle
import os
from constants import *
from tqdm import tqdm
import random

### - Bayes Optimization - ###
from Bayes.Acquisitions import *
from Bayes.BayesOpt_ import *

### - Logging - ###
import tensorboard
from torch.utils.tensorboard import SummaryWriter

### - internal - ###
from pred_network import *
from network_utils import *

"""
Add stratified K means training structure and also bayesopt training cycle

These can come after the RL agent training loop works though
"""

def training_wrapper(bayes, args, epochs, device, loss, model, optim, test=None):
    """
    wrapper function to define and call training structure
    """
    test_scoring = PSNR
    assert(callable(test_scoring))

def PSNR(y_pred, y_true):
    y_pred = y_pred.numpy().astype(np.float64)
    y_true = y_true.numpy().astype(np.float64)
    mse = np.mean((y_pred - y_true)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def static_train(args, epochs, device, loss, model, optim, test=0.1):
    model.train()
    databar = tqdm(range(epochs))
    epoch_losses = []

    ### - Penalty - ###
    g_pen = 0.
    penalty = 0.
    testing = []
    for epoch in databar:
        ### - Databar Init - ###
        losses = []
        databar.set_description('Epoch: %i, Loss: %0.2f, Grad: %0.2f, Regularization Penalty: %.2f, Sample #: %i' % 
        (epoch, 0., 0., 0., 0))
        running_loss = 0.0
        instances = 0
        
        ### - iterate through dataset - ###
        for i, x in enumerate(dataloader):
            x.requires_grad = True
            if x.shape[1] < args.window:
                continue
            x = x.to(device)
            #model.zero_grad()
            optim.zero_grad()
            out = model(x.detach())
            try:
                loss_ = loss(out, x)
                model.eval()
                p = T.norm(model.encoder.get_activations_gradient(), 'fro')
                model.train()
            except:
                p = 0.
            loss_ += p
            loss_.backward()
            
            #losses.append(loss_.item())
            running_loss += np.abs(loss_.item()) / args.batch
            
            databar.set_description('Epoch: %i, Loss: %0.2f, Running Loss: %.2f, Grad Penalty: %e, Sample #: %i' % 
                                    (epoch, loss_.item(), running_loss, p, i))
            if args.clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optim.step()
            if args.sched:
                scheduler.step()
        epoch_losses.append(np.mean(losses))

    print('Saving Trained Network')
    check_point = { 'model' : model.state_dict(), 
                    'optimizer' : optim.state_dict(),
                    'epoch_losses' : epoch_losses,
                           }
    T.save(check_point, 'SM_Representation_Net.pth')
    print('Done!')

def experiment(model, d_keys, args, epochs, device, loss, optim, test_f=0.1):
    score = random.sample(dataset, int(test_f*len(d_keys)))
    static_train(args, epochs, device, loss, model, optim)

    ### - Get output score - ###
    snrs = []
    model.eval()
    for x in score:
        out = model(x.detach())
        snrs.append(PSNR(out.detach(), x.detach()))
    return np.mean(snrs)

def low_cost(model, d_keys, args, epochs, device, loss, optim, test_f=0.1):
    """
    TODO:
        THIS FUNCTION SHOULD ONE RUN EPOCH, TURN THE MODEL TO EVAL, AND RETURN THE MEAN TEST SET SIGNAL TO 
        NOISE RATIO
    """
    score = random.sample(dataset, int(test_f*len(d_keys)))
    model.train()
    databar = tqdm(range(epochs))
    epoch_losses = []

    ### - Penalty - ###
    g_pen = 0.
    penalty = 0.
    for epoch in databar:
        ### - Databar Init - ###
        losses = []
        databar.set_description('Epoch: %i, Loss: %0.2f, Grad: %0.2f, Regularization Penalty: %.2f, Sample #: %i' % 
        (epoch, 0., 0., 0., 0))
        running_loss = 0.0
        instances = 0
        
        ### - iterate through dataset - ###
        for i, x in enumerate(dataloader):
            x.requires_grad = True
            x = x.to(device)
            #model.zero_grad()
            optim.zero_grad()
            out = model(x.detach())
            loss_ = loss(out, x)
            try:
                model.eval()
                p = T.norm(model.encoder.get_activations_gradient(), 'fro')
                model.train()
            except:
                p = 0.
            loss_ += p
            loss_.backward()
            
            #losses.append(loss_.item())
            running_loss += np.abs(loss_.item()) / args.batch
            
            databar.set_description('Epoch: %i, Loss: %0.2f, Running Loss: %.2f, Grad Penalty: %e, Sample #: %i' % 
                                    (epoch, loss_.item(), running_loss, p, i))
            if args.clip:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optim.step()
            if args.sched:
                scheduler.step()
        snrs = []
        model.eval()
        for x in score:
            out = model(x.detach())
            snrs.append(PSNR(out.detach(), x.detach()))
        return np.mean(snrs)
        epoch_losses.append(np.mean(losses))

    print('Saving Trained Network')
    check_point = { 'model' : model.state_dict(), 
                    'optimizer' : optim.state_dict(),
                    'epoch_losses' : epoch_losses,
                           }
    T.save(check_point, 'SM_Representation_Net.pth')
    print('Done!')

def train_self_optimize(args, epochs, device, loss, model, optim):
    ### - Create bayesian framework - ###
    obj_func = None
    if args.bs == 'Experiment':
        obj_func = experiment
    elif args.bs == 'Low Cost':
        obj_func = 2
    else:
        print('invalid Bayesian Optimization structure, please input either Experiment or Low Cost')
        raise ValueError
    
    print('build acquisition function of type %s' % args.bs)
    

if __name__ == '__main__':
    args = process_command_line_arguments()
    #if args.log:
    #    sw = SummaryWriter(args.log_dir)
    
    ### - Get dataset (maybe overwritten)path
    dataset_p_ = None
    if args.in_dest is None:
        dataset_p_ = dataset_path
    else:
        dataset_p_ = args.in_dest
    
    print('Load data from: %s' % dataset_p_)

    ### - training structure args - ###
    debug = args.debug
    optim = args.optim

    if debug:
        print('Training in Debug mode Data will have %i samples' % debug_len)
    if optim:
        print('Use self directed Bayesian Optimization for hyperparameter tuning')

    ### - Create Networks - ###
    lr = args.lr
    epochs = args.epochs
    optim_args = (lr, (0, 0.99))
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    ### - Define Loss - ###
    reg_fn = Regularizer(0.05)
    loss = get_loss_fn(args.loss)
    print(loss)
    assert(callable(loss_fn))

    ### - Create Dataset/DataLoader - ###
    if args.d:
        from data import *
        download_build(dataset_p_)

    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = StockData(dataset_p_, 30, device, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.dw)

    num_features = dataset.features()
    dims = dataset[0].shape
    
    ### - Model Definition - ###
    encoder_args = {'batch_size': args.batch, 'window_size': args.window, 'latent': 138,
                    'dims': dims}
    decoder_args = {}

    model = VGG16_AE(encoder_args, decoder_args, device)

    if not args.adam is None:
        print('Use custom optimizer')
        optim = T.optim.Adam(model.parameters(), *args.adam)
    else:
        print('Default optimizer')
        optim = T.optim.Adam(model.parameters(), *optim_args)

    scheduler = None
    if args.sched:
        scheduler = T.optim.lr_scheduler.ExponentialLR(optim, 0.9, last_epoch=- 1, verbose=False)
    #TODO: Implement Bayes Training cycle
    static_train(args, epochs, device, loss, model, optim)