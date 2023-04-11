import pfrl
import torch
import torch.optim as optim
import numpy as np
from pfrl.agents import A2C
from models import FCN
from config import Args, ActionSpace
from utils import get_device
from data import *


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    # data related stuff
    A = generate_A(args.m, args.n)
    transform = get_transform(args.image_size)
    dataset = MyCSDataset(args.data_dir, A, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    data_iterator = iter(dataloader)  # use iterator to get data

    # calc Qinit
    print("calculating Qinit...")
    Q_init = calc_Qinit(dataloader, device=device)
    print(f"Qinit shape: {Q_init.shape}")

    # define model and other parameters
    actions = ActionSpace()
    model = FCN(action_size=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    lr_lambda = lambda episode: (1 - episode / args.max_episode) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

    # start
    T = 0
    t = 1

    while T <= args.Tmax:
        # initialization
        optimizer.zero_grad()
        loss_theta_p = 0
        loss_theta_v = 0
        loss_w = 0
        t_start = t

        # obtain some data
        try:
            terminal_state, _, state_y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            terminal_state, _, state_y = next(data_iterator)

        state_x = torch.matmul(Q_init, state_y).reshape(-1, 1, args.image_size, args.image_size).to(device)

        while t - t_start < args.tmax:
            policy, value = model(state_x)
            action_idx = policy.sample()
            action = action_idx.clone().detach().cpu()
            action.apply_(lambda x: actions[x])
            
            t += 1
            T += 1
