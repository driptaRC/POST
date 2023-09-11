import argparse 
import random 
import warnings
import os
import sys
import time

import numpy as np 
import torch 
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

from models import  *

sys.path.append('..')
from lib.meter import AverageMeter, ProgressMeter
from lib.data import ForeverDataIterator

def train_model(device, iter_list, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    loss_c = AverageMeter('Loss', ":.4e")

    iters_per_epoch = args.iters_per_epoch
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, loss_c],
        prefix="Epoch: [{}]".format(epoch))   
    
    model.train()
    end = time.time()

    for i in range(iters_per_epoch):
        loss_item = 0.
        for j, iter in enumerate(iter_list):
            x, y = next(iter)
            x = x.to(device)
            y = y.to(device)
            y = torch.unsqueeze(y,1)
            if j==0:
                data_time.update(time.time() - end)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.float())
            loss.backward()
            optimizer.step()
            loss_item += loss.item()
        
        loss_c.update(loss_item, x.size(0)*len(iter_list))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model 
    model = PoseNDF()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    
    # create dataset 
    clean_data = np.load(os.path.join(args.data_root,'real.npy'), allow_pickle=True)
    noisy_data_1 = np.load(os.path.join(args.data_root,'synth_1.npy'), allow_pickle=True)
    noisy_data_2 = np.load(os.path.join(args.data_root,'synth_2.npy'), allow_pickle=True)
    
    x_r = torch.from_numpy(clean_data[()]['poses'])
    y_r = torch.from_numpy(clean_data[()]['labels'])
    x_n_1 = torch.from_numpy(noisy_data_1[()]['poses'])
    y_n_1 = torch.from_numpy(noisy_data_1[()]['labels'])
    x_n_2 = torch.from_numpy(noisy_data_2[()]['poses'])
    y_n_2 = torch.from_numpy(noisy_data_2[()]['labels'])

    dset_r = TensorDataset(x_r,y_r)
    dset_n_1 = TensorDataset(x_n_1,y_n_1)
    dset_n_2 = TensorDataset(x_n_2,y_n_2) 
    loader_r = DataLoader(dset_r, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_n_1 = DataLoader(dset_n_1, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_n_2 = DataLoader(dset_n_2, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True) 

    iter_r = ForeverDataIterator(loader_r)
    iter_n_1 = ForeverDataIterator(loader_n_1)
    iter_n_2 = ForeverDataIterator(loader_n_2)
    
    # STAGE 1
    # train model
    print('Stage 1 training ...')
    for epoch in range(args.epochs_1):
        train_model(device, [iter_n_1], model, optimizer, criterion, epoch, args)
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.out_dir, 'prior_stage_1.pt')
        )
    
    # STAGE 2
    # train model
    print('Stage 2 training ...')
    for epoch in range(args.epochs_2):
        train_model(device, [iter_n_2, iter_n_1], model, optimizer, criterion, epoch, args)
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.out_dir, 'prior_stage_2.pt')
        )

    # STAGE 3
    # train model
    print('Stage 3 training ...')
    for epoch in range(args.epochs_3):
        train_model(device, [iter_r, iter_n_2, iter_n_1], model, optimizer, criterion, epoch, args)
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(args.out_dir, 'prior_stage_3.pt')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train prior')

    parser.add_argument("--data-root", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/Human36M/K_5/processed/',
                        help="file containing clean poses")
    parser.add_argument("--out-dir", type=str, default='/data/AmitRoyChowdhury/dripta/prior_v1/Human36M/K_5/checkpoints/l2/',
                        help="directory to save all poses")
    parser.add_argument("--batch-size",  type=int, default=1024,
                        help='mini-batch size (default: 1024)')
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help='learning rate')
    parser.add_argument("--loss", type=str, default='l2',
                        help="type of loss")
    parser.add_argument("--iters-per-epoch", type=int, default=1000, 
                        help='iterations per epoch')
    parser.add_argument("--epochs-1", type=int, default=75, 
                        help='number of total epochs to run')
    parser.add_argument("--epochs-2", type=int, default=100, 
                        help='number of total epochs to run')
    parser.add_argument("--epochs-3", type=int, default=200, 
                        help='number of total epochs to run')
    parser.add_argument("--seed", type=int, default=0, 
                        help='seed for initializing training. ')
    parser.add_argument("--print-freq", type=int, default=100, 
                        help='print frequency (default: 100)')
    args = parser.parse_args()
    main(args)
