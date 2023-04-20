import torch
import imageio
import numpy as np
from torchvision.utils import save_image


def get_device(device_num):
    if device_num is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_num == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


def get_min_max_data(Qinit, dataloader):
    for _, _, flat_y in dataloader:
        matmul = torch.matmul(Qinit, flat_y)
        min_val = matmul.min()
        max_val = matmul.max()
        break

    for _, _, flat_y in dataloader:
        matmul = torch.matmul(Qinit, flat_y)
        min_val = min(min_val, matmul.min())
        max_val = max(max_val, matmul.max())

    return min_val, max_val


def rescale_tensor_01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def np_to_image_save(np_array, path):
    imageio.imwrite(path, np_array)


def scale_array_uint8(arr, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        return ((arr - arr.min()) * (1 / arr.ptp()) * 255).astype(np.uint8)
    
    return ((arr - min_val) * (1 / (max_val - min_val)) * 255).astype(np.uint8)


def scale_array_float32(arr, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        return ((arr - arr.min()) * (1 / arr.ptp()) * 255).astype(np.float32)
    
    return ((arr - min_val) * (1 / (max_val - min_val)) * 255).astype(np.float32)
