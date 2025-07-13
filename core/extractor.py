import torch
import torch.nn as nn
import torch.nn.functional as F
from core.submodules import DeformConv2d
from core.submodules import conv
from core.SPCII import SPCII


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, batchNorm = True):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.batchNorm = batchNorm

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv_offset1 = conv(self.batchNorm, 128, 18, kernel_size=3, padding=1,
                            dilation=1)  # 18是因为 3*3*2，3*3是卷积核大小，2是x和y方向的偏移
        self.conv2_1 = DeformConv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv_offset2 = conv(self.batchNorm, 128, 18, kernel_size=3, padding=2, dilation=3)
        self.conv2_2 = DeformConv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=3)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_offset3 = conv(self.batchNorm, 128, 18, kernel_size=3, padding=4, dilation=5)
        self.conv2_3 = DeformConv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=5)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(768, output_dim, kernel_size=1, stride=1, padding=0)

        self.spcii = SPCII(output_dim, output_dim, groups=32)

        # output convolution
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x1_offset = self.conv_offset1(x)
        x1 = self.conv2_1(x, x1_offset)
        x1 = self.bn1(x1)
        # print("Size before deformable conv x1:", x1.size())

        x2_offset = self.conv_offset2(x)
        x2 = self.conv2_2(x, x2_offset)
        x2 = self.bn2(x2)
        # print("Size before deformable conv x2:", x2.size())

        x3_offset = self.conv_offset3(x)
        x3 = self.conv2_3(x, x3_offset)
        x3 = self.bn3(x3)
        # print("Size before deformable conv x3:", x3.size())
        # print("x3=",x3)

        # Concatenate the deformable convolution outputs along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)
        # print("xc=",x)

        x = self.conv2(x)
        # print("Size before deformable conv 2:", x.size())
        x = self.spcii(x)
        # print("Size before deformable conv spcii:", x.size())

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0,batchNorm = True):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.batchNorm = batchNorm

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.conv_offset1 = conv(self.batchNorm, 96, 18, kernel_size=3, padding=1,
                                 dilation=1)  # 18是因为 3*3*2，3*3是卷积核大小，2是x和y方向的偏移
        self.conv2_1 = DeformConv2d(96, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_offset2 = conv(self.batchNorm, 96, 18, kernel_size=3, padding=2, dilation=3)
        self.conv2_2 = DeformConv2d(96, 128, kernel_size=3, stride=1, padding=2, dilation=3)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv_offset3 = conv(self.batchNorm, 96, 18, kernel_size=3, padding=4, dilation=5)
        self.conv2_3 = DeformConv2d(96,128, kernel_size=3, stride=1, padding=4, dilation=5)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(384, output_dim, kernel_size=1, stride=1, padding=0)

        self.spcii = SPCII(output_dim, output_dim, groups=32)



        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1_offset = self.conv_offset1(x)
        x1 = self.conv2_1(x, x1_offset)
        x1 = self.bn1(x1)
        # print("Size before deformable conv x1:", x1.size())

        x2_offset = self.conv_offset2(x)
        x2 = self.conv2_2(x, x2_offset)
        x2 = self.bn2(x2)
        # print("Size before deformable conv x2:", x2.size())

        x3_offset = self.conv_offset3(x)
        x3 = self.conv2_3(x, x3_offset)
        x3 = self.bn3(x3)
        # print("Size before deformable conv x3:", x3.size())
        # print("x3=",x3)

        # Concatenate the deformable convolution outputs along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)
        # print("xc=",x)

        x = self.conv2(x)
        # print("Size before deformable conv 2:", x.size())
        x = self.spcii(x)


        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
