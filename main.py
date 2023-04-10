import pfrl
import torch
import torch.optim as optim
import numpy as np
from pfrl.agents import A2C
from models import FCN
from config import Args
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

    # calc Qinit
    print("calculating Qinit...")
    Q_init = calc_Qinit(dataloader, device=device)
    print(f"Qinit shape: {Q_init.shape}")

    # define model and other parameters
    model = FCN(action_size=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    lr_lambda = lambda episode: (1 - episode / args.max_episode) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
