
import torch
import torch.nn.functional as F
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import sys

def train_single_epoch(net, optimizer, train_loader, sigma_n=25, scheduler=None, verbose=False):

    n_data = 0
    epoch_loss = 0.
    net.train()

    if verbose:
        pbar = tqdm(total=len(train_loader), position=0, leave=False, file=sys.stdout)

    for images in train_loader:
        optimizer.zero_grad()
        batch_size = images.size(0)
        images = images.to(net.device)
        y = torch.randn_like(images) * (sigma_n/255.) + images
        noise_hat = net(y)
        x_hat = y - noise_hat
        loss = F.mse_loss(x_hat, images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_size
        n_data += batch_size
        if scheduler is not None:
            scheduler.step()
        if verbose:
            pbar.update(1)

    if verbose:
        tqdm.close(pbar)

    return epoch_loss / float(n_data)

@torch.no_grad()
def validate(net, test_loader, sigma_n=25, verbose=False):

    n_data = 0
    total_loss = 0.
    net.eval()

    if verbose:
        pbar = tqdm(total=len(test_loader), position=0, leave=False, file=sys.stdout)

    for images in test_loader:
        batch_size = images.size(0)
        images = images.to(net.device)
        y = torch.randn_like(images) * (sigma_n/255.) + images
        noise_hat = net(y)
        x_hat = y - noise_hat
        loss = F.mse_loss(x_hat, images)
        total_loss += loss.item() * batch_size
        n_data += batch_size
        if verbose:
            pbar.update(1)

    if verbose:
        tqdm.close(pbar)

    return total_loss / float(n_data)


def train(net, optimizer, max_epoch, train_loader,
          sigma_n=25, validation=None, scheduler=None, lr_step='epoch',
          checkpoint_dir=None, max_tolerance=None, verbose=False):

    best_loss = np.inf
    tolerated = 0

    if lr_step == 'epoch':
        lr_step_per_epoch = True
    elif lr_step == 'batch':
        lr_step_per_epoch = False
    else:
        lr_step_per_epoch = True

    epoch_scheduler = None
    if not lr_step_per_epoch:
        epoch_scheduler = scheduler

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        if verbose:
            print('\nEpoch #{:d}'.format(e+1))

        log[e, 0] = train_single_epoch(net, optimizer, train_loader, sigma_n=sigma_n, scheduler=epoch_scheduler, verbose=verbose)

        if verbose:
            print('Train Loss: {:.5f}'.format(log[e, 0]))

        if (scheduler is not None) and lr_step_per_epoch:
            scheduler.step()

        if validation is not None:

            log[e, 1] = validate(net, validation, sigma_n, verbose=verbose)

            if verbose:
                print('Val Loss: {:.5f}'.format(log[e, 1]))

            if (checkpoint_dir is not None) and (best_loss > log[e, 1]):
                best_loss = log[e, 1]
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint'+str(e+1)+'.pth')
                torch.save(net.state_dict(), checkpoint_path)
                if verbose:
                    print('Best Loss! Saved.')
            elif max_tolerance is not None:
                tolerated += 1
                if tolerated > max_tolerance:
                    return log[0:e, :]

    return log
