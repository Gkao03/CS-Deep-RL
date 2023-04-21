import torch
import numpy as np
import cv2


class BoxFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return cv2.boxFilter(img, -1, (self.kernel_size, self.kernel_size))
    
    def __repr__(self):
        return f"Box({self.kernel_size})"
    

class BilateralFilter:
    def __init__(self, kernel_size, sigma_color, sigma_space):
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img):
        return cv2.bilateralFilter(img, self.kernel_size, self.sigma_color, self.sigma_space)
    
    def __repr__(self):
        return f"Bilat({self.kernel_size}, {self.sigma_color}, {self.sigma_space})"
    

class MedianFilter:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return cv2.medianBlur(img, self.kernel_size)
    
    def __repr__(self):
        return f"Median({self.kernel_size})"
    

class GaussianFilter:
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigmaX=self.sigma, sigmaY=self.sigma)
    
    def __repr__(self):
        return f"Gauss({self.kernel_size}, {self.sigma})"
    

class IncrementValue:
    def __init__(self, value):
        self.value = value

    def __call__(self, img):
        return img + self.value
    
    def __repr__(self):
        return f"Inc({self.value})"
    

class DoNothing:
    def __init__(self):
        pass

    def __call__(self, img):
        return img
    
    def __repr__(self):
        return "DoNothing()"


def vec_apply(action_idx, x, y, action_map):
    return action_map[action_idx][x, y]


class ApplyAction:
    def __init__(self, actions):
        self.actions = actions
        self.vec_apply = np.vectorize(vec_apply, excluded=['action_map'])

    def __call__(self, curr_state, action_idx):
        curr_state = curr_state * 255
        next_states = []
        m, n = curr_state.shape[-2:]
        y, x = np.meshgrid(np.arange(n), np.arange(m))

        # split curr_state
        imgs = torch.split(curr_state, 1, dim=0)
        imgs = [img.squeeze().numpy() for img in imgs]

        # split action idxs
        action_idxs = torch.split(action_idx, 1, dim=0)
        action_idxs = [idx.squeeze().numpy() for idx in action_idxs]  # idx representing actions

        for i, (img, action_idx) in enumerate(zip(imgs, action_idxs)):
            action_map = dict()

            for key, fn in self.actions.items():
                action_map[key] = fn(img)

            next_state = self.vec_apply(action_idx, x, y, action_map)
            next_states.append(torch.tensor(next_state, dtype=torch.float32))

        next_state_tensor = torch.stack(next_states, dim=0)
        next_state_tensor = torch.unsqueeze(next_state_tensor, dim=1)

        return next_state_tensor / 255
