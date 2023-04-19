import torch
import numpy as np
import cv2


class ApplyAction:
    def __init__(self, actions):
        self.actions = actions

    def __call__(self, curr_state, action_idx):
        next_states = [torch.zeros(curr_state.shape[-2:]) for _ in range(curr_state.shape[0])]

        # split curr_state
        imgs = torch.split(curr_state, 1, dim=0)
        imgs = [img.squeeze().numpy() for img in imgs]

        # split action idxs
        action_idxs = torch.split(action_idx, 1, dim=0)

        for i, (img, action_idx) in enumerate(zip(imgs, action_idxs)):
            action = self.actions[int(action_idx)]
            next_states[i] = img * action
