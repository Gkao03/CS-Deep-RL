import pfrl
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pfrl.agents import A2C
from models import FCN, RewardConv
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
    actions = ActionSpace().action_space
    model = FCN(action_size=len(actions)).to(device)
    reward_conv = RewardConv(args.w_filter_size).to(device)
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
            target_state, _, state_y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            target_state, _, state_y = next(data_iterator)

        curr_state = torch.matmul(Q_init, state_y).reshape(-1, 1, args.image_size, args.image_size)

        # saved output
        policies = []
        values = []
        rewards = []

        while t - t_start < args.tmax:
            # curr_state
            curr_state = curr_state.to(device)

            # feed through network
            policy, value = model(curr_state)

            # sample and get action
            action_idx = policy.sample()
            action = action_idx.clone().detach().cpu().float()
            action.apply_(lambda x: actions[int(x)])

            # get next_state
            next_state = curr_state.detach().cpu() * action

            # calculate reward
            reward = torch.square(target_state - curr_state) - torch.square(target_state - next_state)

            # append
            policies.append(policy)
            values.append(value)
            rewards.append(reward)

            # update counters
            t += 1
            T += 1

        # get reward map
        R = value.detach()

        for pi, V, r in zip(reversed(policies), reversed(values), reversed(rewards)):
            R = r + args.gamma * R
            loss_theta_p += pi.log_prob(action_idx) * (R - V)
            loss_theta_v += F.smooth_l1_loss(V, R)
            loss_w += F.smooth_l1_loss(reward_conv(curr_state), R)
