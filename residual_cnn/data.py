
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import sys


def get_transform(mode):
    if mode.lower() == 'train':
        T = transforms.Compose([transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

    elif mode.lower() in ['val', 'test']:
        T = transforms.Compose([transforms.ToTensor()])

    elif mode.lower() == 'none':
        T = transforms.Compose([])

    return T

class Rescale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        new_size = (int(x.size[0] * self.scale), int(x.size[1] * self.scale))
        return transforms.Resize(new_size)(x)

class ImageDataSubset(Dataset):
    def __init__(self, dataset, indices, mode='none', patch_size=None, repeat=1):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.repeat = repeat
        self.patch_size = patch_size
        self.transform = get_transform(mode)
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if self.patch_size is not None:
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))

    def set_patch(self, patch_size):
        if self.patch_size is not None:
            self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __getitem__(self, idx):
        return self.transform(self.dataset.images[self.indices[idx // self.repeat]])

    def __len__(self):
        return len(self.indices) * self.repeat

class ImageDataset(Dataset):
    def __init__(self, root_dirs, mode='none', gray=True, rescale=None,
                 patch_size=None, repeat=1, extensions='png'):
        super().__init__()
        self.images = list()
        self.repeat = repeat
        self.rescale = rescale
        self.patch_size = patch_size
        self.gray = gray

        if type(extensions) != list:
            extensions = [extensions]

        if type(root_dirs) != list:
            root_dirs = [root_dirs]

        file_paths = list()
        for root_dir in root_dirs:
            for ext in extensions:
                file_paths += glob.glob(os.path.join(root_dir, '*.'+ext))
        n_files = len(file_paths)

        pbar = tqdm(total=n_files, position=0, leave=False, file=sys.stdout)

        for file_path in file_paths:
            fptr = Image.open(file_path)
            if self.gray:
                fptr = fptr.convert('L')
            else:
                fptr = fptr.convert('RGB')
            file_copy = fptr.copy()
            fptr.close()
            if rescale is not None:
                file_copy = Rescale(scale)(file_copy)
            self.images.append(file_copy)
            pbar.update(1)

        tqdm.close(pbar)

        self.transform = get_transform(mode)
        if self.patch_size is not None:
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))

    def set_mode(self, mode):
        self.transform = get_transform(mode)
        if (self.patch_size is not None):
            self.transform.transforms.insert(0, transforms.RandomCrop(self.patch_size))

    def set_patch(self, patch_size):
        if self.patch_size is not None:
            self.transform.transforms.pop(0)
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.transform.transforms.insert(0, transforms.RandomCrop(patch_size))

    def __len__(self):
        return int(len(self.images) * self.repeat)

    def __getitem__(self, idx):
        return self.transform(self.images[idx // self.repeat])

    def split(self, *r):
        ratios = np.array(r)
        ratios = ratios / ratios.sum()
        total_num = len(self.images)
        indices = np.arange(total_num)
        np.random.shuffle(indices)

        subsets = list()
        start = 0
        for r in ratios[:-1]:
            split = int(total_num * r)
            subsets.append(ImageDataSubset(self, indices[start:start+split]))
            start = start + split
        subsets.append(ImageDataSubset(self, indices[start:]))

        return subsets
