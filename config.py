import os

class Args:
    def __init__(self):
        self.seed = 40
        self.data_dir = "data/pristine_images"
        self.tmax = 5
        self.max_episode = 30000
        self.Tmax = self.tmax * self.max_episode
        self.w_filter_size = 33
        self.lr_init = 1e-3
        self.image_size = 64
        self.gamma = 0.9
        self.n = self.image_size ** 2
        self.m = int(0.2 * self.n)
        self.batch_size = 64
        self.exp_num = 3
        self.out_dir = os.path.join("out", f"exp{self.exp_num}")
        self.log_step = 100


class ActionSpace:
    def __init__(self):
        # idx to action (multiply)
        self.action_space = {0: 0.3,
                             1: 0.5,
                             2: 0.7,
                             3: 0.9,
                             4: 1.0,
                             5: 1.1,
                             6: 1.3,
                             7: 1.5,
                             8: 1.7,
                             9: 2.0}
