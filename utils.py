import torch
import imageio
from torchvision.utils import save_image
from skimage.util import random_noise


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def rescale_tensor_01(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)


def np_to_image_save(np_array, path):
    imageio.imwrite(path, np_array)


def scale_array_uint8(arr):
    new_arr = ((arr - arr.min()) * (1 / arr.ptp() * 255)).astype('uint8')
    return new_arr
