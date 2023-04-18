import os

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
        self.exp_num = 9
        self.out_dir = os.path.join("out", f"exp{self.exp_num}")
        self.log_step = 100
        self.transform_method = "dct"
        self.A_path = os.path.join(self.out_dir, "A.npy")
        self.Qinit_path = os.path.join(self.out_dir, "Q_init.npy")
        self.device_num = None


class ActionSpace:
    def __init__(self):
        # idx to action (multiply)
        self.action_space = {0: 0.9,
                             1: 0.95,
                             2: 1.0,
                             3: 1.05,
                             4: 1.1,}
