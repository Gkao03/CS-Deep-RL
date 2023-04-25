import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import save_image
from models import FCN, RewardConv
from config import Args, ActionSpace
from utils import get_device, get_min_max_data, rescale_tensor_01, np_to_image_save, scale_array_uint8
from actions import ApplyAction
from data import *


# visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def reconstruct_CS(model, reward_conv, A, Q_init, tmax, dataloader, apply_action, device, out_dir):
    A = torch.tensor(A, dtype=torch.float32).to(device)
    model = model.to(device)
    reward_conv = reward_conv.to(device)
    model.eval()
    reward_conv.eval()

    # just get 1 image from dataloader
    for target_state, _, state_y in dataloader:
        curr_state = torch.matmul(Q_init, state_y).reshape(-1, 1, args.image_size, args.image_size)
        save_image(curr_state, os.path.join(out_dir, "before_recon.png"))
        state_y = state_y.to(device)

        for _ in range(tmax):
            curr_state = curr_state.to(device)

            # gradient step
            gradient = A.t() @ (A @ curr_state.reshape(-1, 1, args.n, 1) - state_y)
            curr_state = curr_state - args.eta * gradient.reshape(-1, 1, args.image_size, args.image_size)

            # feed through network
            policy, _ = model(curr_state)

            # sample and get action
            action_idx = policy.sample()
            action = action_idx.clone().detach().cpu().float()
            # action.apply_(lambda x: actions[int(x)])
            # action = torch.unsqueeze(action, dim=1)

            # get next_state
            next_state = apply_action(curr_state.detach().cpu(), action)
            curr_state = next_state

        # save images
        original = target_state.detach().squeeze()
        # original = scale_array_uint8(original)
        reconstructed = curr_state.detach().squeeze()
        # reconstructed = scale_array_uint8(reconstructed)

        # np_to_image_save(original, os.path.join(out_dir, "original.png"))
        # np_to_image_save(reconstructed, os.path.join(out_dir, "reconstructed.png"))
        save_image(original, os.path.join(out_dir, "original.png"))
        save_image(reconstructed, os.path.join(out_dir, "reconstructed.png"))
        break

    print("DONE")


def reconstruct_denoise(model, reward_conv, tmax, dataloader, apply_action, device, out_dir, num_images=1):
    img_counter = 0
    model = model.to(device)
    reward_conv = reward_conv.to(device)
    model.eval()
    reward_conv.eval()


    for target_state, curr_state in dataloader:
        if img_counter >= num_images:
            break

        save_image(curr_state, os.path.join(out_dir, "noisy.png"))

        for _ in range(tmax):
            curr_state = curr_state.to(device)

            # feed through network
            policy, _ = model(curr_state)

            # sample and get action
            action_idx = policy.sample()
            action = action_idx.clone().detach().cpu().float()
            # action.apply_(lambda x: actions[int(x)])
            # action = torch.unsqueeze(action, dim=1)

            # get next_state
            # next_state = curr_state.detach().cpu() * action
            next_state = apply_action(curr_state.detach().cpu(), action)
            curr_state = next_state

        # save images
        original = target_state.detach().squeeze()
        # original = scale_array_uint8(original)
        reconstructed = curr_state.detach().squeeze()
        # reconstructed = scale_array_uint8(reconstructed)

        # np_to_image_save(original, os.path.join(out_dir, "original.png"))
        # np_to_image_save(reconstructed, os.path.join(out_dir, "reconstructed.png"))
        save_image(original, os.path.join(out_dir, "original.png"))
        save_image(reconstructed, os.path.join(out_dir, "reconstructed.png"))

        img_counter += 1

    print("DONE")


if __name__ == "__main__":
    args = Args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device_num)

    # data related stuff (CS)
    # A = np.load(args.A_path)
    # transform = get_transform(args.image_size)
    # dataset = MyCSDataset(args.data_dir, A, transform=transform)
    # qinit_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # get A (CS)
    print("getting A...")
    A = np.load(os.path.join(args.out_dir, "A.npy"))

    # data related stuff (denoising)
    transform = get_transform(args.image_size, train=True)  # train False for denoise/ train True for CS
    # dataset = MyNoisyDataset(args.data_dir, transform=transform)  # denoise
    dataset = MyCSDataset(args.data_dir, A, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # calc Qinit (CS)
    print("getting Qinit...")
    Q_init = torch.tensor(np.load(os.path.join(args.out_dir, "Qinit.npy")))
    print(f"Qinit shape: {Q_init.shape}")

    # get min and max
    # min_val, max_val = get_min_max_data(Q_init, qinit_dataloader)

    # define models
    actions = ActionSpace().action_space
    model = FCN(action_size=len(actions)).to(device)
    reward_conv = RewardConv(args.w_filter_size).to(device)
    apply_action = ApplyAction(actions)

    # load state dicts from trained models
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "model.pth")))
    reward_conv.load_state_dict(torch.load(os.path.join(args.out_dir, "reward_conv.pth")))
    print("state dicts loaded")

    # dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # call reconstruction (CS)
    reconstruct_CS(model, reward_conv, A, Q_init, args.tmax, dataloader, apply_action, device, args.out_dir)
    # reconstruct_denoise(model, reward_conv, args.tmax, dataloader, apply_action, device, args.out_dir)
