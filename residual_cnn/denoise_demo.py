
from PIL import Image
import numpy as np
import torch
import argparse
import copy

import model
import data

parser = argparse.ArgumentParser()
parser.add_argument('-model_path', type=str, default='weights/mydenoiser_041923182445_nobn.pth')
parser.add_argument('-image', type=str, default='../exploration_database_and_code/pristine_images/00004.bmp')
args = parser.parse_args()

img = Image.open(args.image).convert('L')
T = data.get_transform('test')

net = model.ResidualCNN_noBN()
net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
net.eval()

with torch.no_grad():
    x = T(img)
    y = x + torch.randn_like(x) * 25. / 255
    n_hat = net(y.unsqueeze(0))
    x_hat = y - n_hat

noisy = y.squeeze().numpy()
recon = x_hat.squeeze().numpy()

noisy = (255*np.clip(noisy, 0, 1)).astype(np.uint8)
recon = (255*np.clip(recon, 0, 1)).astype(np.uint8)

img = np.array(img)
h, w = img.shape
spacing = np.zeros((h, 20), dtype=np.uint8)
stacked_img = np.concatenate([img, spacing, noisy, spacing, recon], axis=1)
Image.fromarray(stacked_img, mode='L').show()
