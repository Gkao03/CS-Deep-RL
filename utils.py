import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_min_max_data(Qinit, dataloader):
    for _, _, flat_y in dataloader:
        min_val = flat_y.min()
        max_val = flat_y.max()
        break

    for _, _, flat_y in dataloader:
        min_val = min(min_val, flat_y.min())
        max_val = max(max_val, flat_y.max())

    return min_val, max_val
