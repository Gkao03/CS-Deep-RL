
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sigma_n', type=float, default=25.)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-patch_size', type=int, default=64)
    parser.add_argument('-rescale', type=int, default=None)
    parser.add_argument('-train_data_dirs', type=str, nargs='+', default=['DIV2K_train_HR'])
    parser.add_argument('-test_data_dirs', type=str, nargs='+', default=['DIV2K_valid_HR'])
    parser.add_argument('-validate', action='store_true')
    parser.add_argument('-n_epoch', type=int, default=100)
    parser.add_argument('-nogpu', action='store_true')
    parser.add_argument('-name', type=str, default='mydenoiser')
    parser.add_argument('-verbose', action='store_true')
    return parser.parse_args()
