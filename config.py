import os
from actions import *

class Args:
    def __init__(self):
        self.seed = 40
        self.data_dir = "data/pristine_images"
        self.tmax = 5
        self.max_episode = 20000  # 30000
        self.Tmax = self.tmax * self.max_episode
        self.w_filter_size = 33
        self.lr_init = 1e-3
        self.image_size = 64
        self.gamma = 0.9
        self.n = self.image_size ** 2
        self.m = int(0.2 * self.n)
        self.batch_size = 64
        self.exp_num = 12
        self.out_dir = os.path.join("out", f"exp{self.exp_num}")
        self.log_step = 100
        self.transform_method = "dct"
        self.A_path = os.path.join(self.out_dir, "A.npy")
        self.Qinit_path = os.path.join(self.out_dir, "Q_init.npy")
        self.device_num = None


class ActionSpace:
    def __init__(self):
        # idx to action (multiply)
        # self.action_space = {0: 0.7,
        #                      1: 0.8,
        #                      2: 0.9,
        #                      3: 1.1,
        #                      4: 1.2,
        #                      5: 1.3}
        
        # new action space
        self.action_space = {0: BoxFilter(5),
                             1: BilateralFilter(5, 1.0, 5.0),
                             2: BilateralFilter(5, 0.1, 5.0),
                             3: MedianFilter(5),
                             4: GaussianFilter(5, 1.5),
                             5: GaussianFilter(5, 0.5),
                             6: IncrementValue(1.0),
                             7: IncrementValue(-1.0),
                             8: DoNothing(),
        }
