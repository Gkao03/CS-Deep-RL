
import torch
from torch import nn
import model


def collapse_bn(net1):

    net1.eval()
    net2 = model.ResidualCNN_noBN()

    net1_param = dict(net1.named_children())
    net2_param = dict(net2.named_children())

    net1_conv = [net1_param['conv{:d}'.format(i)] for i in range(1,8)]
    net1_bn = [net1_param['bn{:d}'.format(i)] for i in range(1,7)]
    net2_conv = [net2_param['conv{:d}'.format(i)] for i in range(1,8)]

    for c1, c2 in zip(net1_conv, net2_conv):
        c2.weight.data.copy_(c1.weight.data)

    for c2 in net2_conv:
        c2.bias.data.zero_()

    for c2, b1 in zip(net2_conv[:-1], net1_bn):
        invstd = b1.running_var.clone().add_(b1.eps).pow_(-0.5)
        c2.weight.data.mul_(invstd.view(c2.weight.data.size(0), 1, 1, 1).expand_as(c2.weight.data))
        c2.bias.data.add_(-b1.running_mean).mul_(invstd)
        if b1.affine:
            c2.weight.data.mul_(b1.weight.data.view(c2.weight.data.size(0), 1, 1, 1).expand_as(c2.weight.data))
            c2.bias.data.mul_(b1.weight.data).add_(b1.bias.data)


    return net2


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default='weights/mydenoiser_041923182445.pth')
    args = parser.parse_args()

    net = model.ResidualCNN()
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    net2 = collapse_bn(net)

    torch.save(net2.state_dict(), args.model_path.rstrip('.pth')+'_nobn.pth')
