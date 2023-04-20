import torch
import torch.nn as nn
import torch.nn.functional as F
from convgru import ConvGRU
import residual_cnn.model


class SharedNet(nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', dilation=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=2)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=3)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=4)
        self.act4 = nn.ReLU()


    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))

        return out


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super(PolicyNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=3)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=2)
        self.act2 = nn.ReLU()

        self.conv3 = ConvGRU(input_size=64, hidden_sizes=64, kernel_sizes=3, n_layers=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=action_size, kernel_size=3, padding='same', dilation=1)
        self.act4 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.conv3(out)[0]  # returns list. using 1 layer so take index 0
        out = self.act4(self.conv4(out))

        return out


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=3)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dilation=2)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding='same', dilation=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.conv3(out)

        return out


class FCN(nn.Module):
    def __init__(self, action_size):
        super(FCN, self).__init__()

        self.shared_net = SharedNet()
        self.policy_net = PolicyNet(action_size)
        self.value_net = ValueNet()

    def forward(self, x):
        out = self.shared_net(x)
        policy = self.policy_net(out)
        value = self.value_net(out)

        policy_permute = torch.permute(policy, (0, 2, 3, 1))
        return torch.distributions.Categorical(policy_permute), value


class RewardConv(nn.Module):
    def __init__(self, kernel_size):
        super(RewardConv, self).__init__()
        self.reward_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same', dilation=1)

        # init weight
        nn.init.zeros_(self.reward_conv.weight)
        self.reward_conv.weight.data[:, :, (kernel_size + 1) // 2, (kernel_size + 1) // 2] = 1

    def forward(self, R):
        out = self.reward_conv(R)
        return out


# intialize FCN in place with trained residual_cnn
# copy weights except for GRU and the last layers of the policy and value network
def initialize_FCN(fcn, dncnn_path='residual_cnn/weights/mydenoiser_041923182445_nobn.pth'):
    dncnn = residual_cnn.model.ResidualCNN_noBN()
    dncnn.load_state_dict(torch.load(dncnn_path, map_location=torch.device('cpu')))

    shared_net_param = dict(fcn.shared_net.named_children())
    policy_net_param = dict(fcn.policy_net.named_children())
    value_net_param = dict(fcn.value_net.named_children())
    dncnn_param = dict(dncnn.named_children())

    # initliaze shared net
    shared_net_param['conv1'].weight.data.copy_(dncnn_param['conv1'].weight.data)
    shared_net_param['conv1'].bias.data.copy_(dncnn_param['conv1'].bias.data)

    shared_net_param['conv2'].weight.data.copy_(dncnn_param['conv2'].weight.data)
    shared_net_param['conv2'].bias.data.copy_(dncnn_param['conv2'].bias.data)

    shared_net_param['conv3'].weight.data.copy_(dncnn_param['conv3'].weight.data)
    shared_net_param['conv3'].bias.data.copy_(dncnn_param['conv3'].bias.data)

    shared_net_param['conv4'].weight.data.copy_(dncnn_param['conv4'].weight.data)
    shared_net_param['conv4'].bias.data.copy_(dncnn_param['conv4'].bias.data)

    # intialize policy net
    policy_net_param['conv1'].weight.data.copy_(dncnn_param['conv5'].weight.data)
    policy_net_param['conv1'].bias.data.copy_(dncnn_param['conv5'].bias.data)

    policy_net_param['conv2'].weight.data.copy_(dncnn_param['conv6'].weight.data)
    policy_net_param['conv2'].bias.data.copy_(dncnn_param['conv6'].bias.data)

    # intialize value net
    value_net_param['conv1'].weight.data.copy_(dncnn_param['conv5'].weight.data)
    value_net_param['conv1'].bias.data.copy_(dncnn_param['conv5'].bias.data)

    value_net_param['conv2'].weight.data.copy_(dncnn_param['conv6'].weight.data)
    value_net_param['conv2'].bias.data.copy_(dncnn_param['conv6'].bias.data)

    return
