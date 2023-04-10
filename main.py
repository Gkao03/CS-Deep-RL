import pfrl
import torch
import numpy as np
from pfrl.agents import A2C
from models import FCN
from config import Args
from data import *


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)
    np.seed(args.seed)

    x = torch.randn(1, 1, 64, 64)
    model = FCN(action_size=6)
    policy, value = model(x)

    print(policy.shape)
    print(value.shape)
