import torch
import numpy as np
import cv2


class BoxFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return cv2.boxFilter(img, -1, (self.kernel_size, self.kernel_size))
    

class BilateralFilter:
    def __init__(self, kernel_size, sigma_color, sigma_space):
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img):
        return cv2.bilateralFilter(img, self.kernel_size, self.sigma_color, self.sigma_space)
    

class MedianFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return cv2.medianBlur(img, self.kernel_size)
    

class GaussianFilter:
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigmaX=self.sigma, sigmaY=self.sigma)
    

class IncrementValue:
    def __init__(self, value):
        self.value = value

    def __call__(self, img):
        return img + self.value
    

class DoNothing:
    def __init__(self):
        pass

    def __call__(self, img):
        return img
    

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
