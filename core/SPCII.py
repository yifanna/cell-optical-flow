import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['mbv2_ca']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SPCII(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(SPCII, self).__init__()
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)

        self.relu = h_swish()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv4 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def k_size(self, mip):
        b = 1
        gama = 2
        k_size = int(abs((math.log(mip, 2) + b) / gama))
        return k_size - 1

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h_a = self.pool_h_avg(x)  # 输出 （n c none, 1）
        x_h_m = self.pool_h_max(x)  #
        x_w_a = self.pool_w_avg(x).permute(0, 1, 3, 2)  # (n, c, large_value, 1)
        x_w_m = self.pool_w_max(x).permute(0, 1, 3, 2)

        y_h = torch.cat([x_h_a, x_w_a], dim=2)  # (n, c, desired_height, 2)
        y_h = self.conv1(y_h)
        y_h = self.bn1(y_h)
        y_h = self.relu(y_h)

        y_w = torch.cat([x_w_m, x_h_m], dim=2)  #
        y_w = self.conv1(y_w)
        y_w = self.bn1(y_w)
        y_w = self.relu(y_w)

        x_h_a, x_w_a = torch.split(y_h, [h, w], dim=2)
        x_w_a = x_w_a.permute(0, 1, 3, 2)

        x_h_m, x_w_m = torch.split(y_w, [h, w], dim=2)
        x_w_m = x_w_m.permute(0, 1, 3, 2)

        y_h = x_h_a + x_h_m
        # y_h = x_h_a
        y_w = x_w_a + x_w_m
        # y_w =  x_w_a

        new_length = y_h.size(2) * y_h.size(3)  # y_h
        output_1d_h = y_h.view(y_h.size(0), y_h.size(1), new_length)
        in_channels = y_h.size(1)

        conv1d = nn.Conv1d(in_channels, in_channels, self.k_size(in_channels)).to(output_1d_h.device)

        y_h = conv1d(output_1d_h).unsqueeze(-1)

        new_length = y_w.size(2) * y_w.size(3)  # y_w
        output_1d_w = y_w.view(y_w.size(0), y_w.size(1), new_length)
        in_channels = y_w.size(1)

        conv1d = nn.Conv1d(in_channels, in_channels, self.k_size(in_channels)).to(output_1d_w.device)

        y_w = conv1d(output_1d_w).unsqueeze(-2)

        y_h = self.conv4(y_h).sigmoid()
        y_w = self.conv5(y_w).sigmoid()
        y_h = y_h.expand(-1, -1, h, w)
        y_w = y_w.expand(-1, -1, h, w)

        y = identity * y_h * y_w

        return y

