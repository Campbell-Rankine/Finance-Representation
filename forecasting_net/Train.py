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

### - Logging - ###
import tensorboard
from torch.utils.tensorboard import SummaryWriter

### - internal - ###
from pred_network import *
from network_utils import *

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
    assert(callable(loss_fn))

    ### - Create Dataset/DataLoader - ###
    if args.d:
        from data import *
        download_build(dataset_p_)

    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = StockData(dataset_p_, 128, device, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.dw)

    num_features = dataset.features()
    dims = dataset[0].shape
    
    ### - Model Definition - ###
    encoder_args = {'batch_size': args.batch, 'window_size': args.window, 'latent': args.latent,
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
        epoch_losses.append(np.mean(losses))

    print('Saving Trained Network')
    check_point = { 'model' : model.state_dict(), 
                    'optimizer' : optim.state_dict(),
                    'epoch_losses' : epoch_losses,
                           }
    T.save(check_point, 'SM_Representation_Net.pth')
    print('Done!')


