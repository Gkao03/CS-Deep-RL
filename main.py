import time
import os
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from pfrl.agents import A2C
from models import FCN, RewardConv
from config import Args, ActionSpace
from utils import get_device, get_min_max_data, rescale_tensor_01
from data import *

# visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':
    args = Args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device_num)

    # create output dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"created output dir: {args.out_dir}")

    # data related stuff (CS)
    # A = generate_A(args.m, args.n)
    # transform = get_transform(args.image_size)
    # dataset = MyCSDataset(args.data_dir, A, transform=transform)
    # qinit_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # data_iterator = iter(dataloader)  # use iterator to get data

    # data related stuff (denoising)
    transform = get_transform(args.image_size)
    dataset = MyNoisyDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    data_iterator = iter(dataloader)  # use iterator to get data

    # start timer
    start_time = time.time()

    # calc Qinit
    # print("calculating Qinit...")
    # Q_init = calc_Qinit(qinit_dataloader, device=device)
    # print(f"Qinit shape: {Q_init.shape}")

    # save A and Qinit
    # np.save(args.A_path, A)
    # np.save(args.Qinit_path, Q_init.numpy())

    # define model and other parameters
    actions = ActionSpace().action_space
    model = FCN(action_size=len(actions)).to(device)
    reward_conv = RewardConv(args.w_filter_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init)
    lr_lambda = lambda episode: (1 - episode / args.max_episode) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=False)

    # get min and max
    # min_val, max_val = get_min_max_data(Q_init, dataloader)

    # start
    T = 0
    t = 1
    print("start training...")

    while T < args.Tmax:
        # initialization
        optimizer.zero_grad()
        loss_theta_p = 0
        loss_theta_v = 0
        loss_w = 0
        t_start = t

        # obtain some data (CS)
        # try:
        #     target_state, _, state_y = next(data_iterator)
        # except StopIteration:
        #     data_iterator = iter(dataloader)
        #     target_state, _, state_y = next(data_iterator)

        # CS
        # curr_state = torch.matmul(Q_init, state_y).reshape(-1, 1, args.image_size, args.image_size)
        # curr_state = rescale_tensor_01(curr_state, min_val, max_val)

        # obtain some data (denoising)
        try:
            target_state, curr_state = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            target_state, curr_state = next(data_iterator)

        # saved output
        policies = []
        action_idxs = []
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
            action = torch.unsqueeze(action, dim=1)

            # get next_state
            next_state = curr_state.detach().cpu() * action

            # calculate reward
            reward = torch.square(target_state - curr_state.cpu()) - torch.square(target_state - next_state)

            # append
            policies.append(policy)
            action_idxs.append(action_idx)
            values.append(value)
            rewards.append(reward)

            # update counters and curr state
            t += 1
            T += 1
            curr_state = next_state

        # get reward map
        R = value  # keep gradient?

        # iterate backwards
        for pi, act_idx, V, r in reversed(list(zip(policies, action_idxs, values, rewards))):
            # reward map
            r = r.to(device)
            R = args.gamma * R
            R = reward_conv(R)
            R = r + R

            # update losses
            # loss_theta_p += -torch.mean(torch.mean(pi.log_prob(act_idx) * (R - V), dim=(1, 2)))
            # loss_theta_v += F.mse_loss(R, V)
            # loss += loss_theta_p + loss_theta_v

            # update losses try grad accumulation
            # loss_theta_p = -torch.mean(torch.mean(pi.log_prob(act_idx) * (R - V), dim=(1, 2)))
            # loss_theta_v = F.mse_loss(R, V)
            # loss = (loss_theta_p + loss_theta_v) / args.tmax  # normalize by num accumulation steps
            # loss.backward()

            # update losses - grad accumulation for each network individually (exp 8)
            loss_theta_p += -torch.mean(torch.mean(pi.log_prob(act_idx) * (R.detach() - V.detach()), dim=(1, 2)))
            loss_theta_v += F.mse_loss(R.detach(), V)
            loss_w += -torch.mean(torch.mean(pi.log_prob(act_idx).detach() * (R - V.detach()), dim=(1, 2))) + F.mse_loss(R, V.detach())

        # calc gradients and step with optimizer
        loss = (loss_theta_p + loss_theta_v + loss_w) / args.tmax
        loss.backward()
        optimizer.step()
        
        # step scheduler
        scheduler.step()

        # print logging info
        if T % args.log_step == 0 or T == args.tmax:
            print(f"T: {T}, loss: {loss.item()}, loss_theta_p: {loss_theta_p.item()}, loss_theta_v: {loss_theta_v.item()}, loss_w: {loss_w.item()}")

    # end timer
    end_time = time.time()
    print(f"total time: {(end_time - start_time) / 3600} hours")

    # save models
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    print("saved model")
    torch.save(reward_conv.state_dict(), os.path.join(args.out_dir, "reward_conv.pth"))
    print("saved reward_conv")
