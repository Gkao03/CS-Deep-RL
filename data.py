import scipy.linalg as linalg
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def generate_A(m, n):
    assert m <= n, "m should be less than equal to n"

    dft_mat = linalg.dft(n) / np.sqrt(n)
    random_select_row = np.random.permutation(n)
    A = dft_mat[random_select_row[:m], :].real

    return A
