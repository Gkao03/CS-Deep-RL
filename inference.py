import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from models import FCN, RewardConv
from config import Args, ActionSpace
from utils import get_device, get_min_max_data, rescale_tensor_01
from data import *


if __name__ == "__main__":
    args = Args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    # data related stuff
    A = generate_A(args.m, args.n)
    transform = get_transform(args.image_size)
    dataset = MyCSDataset(args.data_dir, A, transform=transform)
    qinit_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # calc Qinit
    print("calculating Qinit...")
    Q_init = calc_Qinit(qinit_dataloader, device=device)
    print(f"Qinit shape: {Q_init.shape}")

    # get min and max
    min_val, max_val = get_min_max_data(Q_init, qinit_dataloader)
