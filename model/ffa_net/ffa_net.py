import torch
import torch.nn as nn
import torch.nn.functional as F

class PALayer(nn.Module):
    """Pixel Attention Layer"""
    def __init__(self, channels):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding = 0, bias = True),
            nn.ReLU(inplace = True), 
            nn.Conv2d(channels // 8, 1, 1, padding = 0, bias = True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, channels, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y
        

class BasicBlock(nn.Module):
    """Basic Block with Channel and Pixel Attention"""
    def __init__(self, channels, kernel_size = 3, reduction = 8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding = kernel_size // 2)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding = kernel_size // 2)

        self.ca = CALayer(channels, reduction)
        self.pa = PALayer(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        out = self.pa(out)
        out += residual 
        
        return out

class Group(nn.Module):
    def __init__(self, channels, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [BasicBlock(channels, kernel_size) for _ in range(blocks)]
        modules.append(
            nn.Conv2d(channels, channels, kernel_size, padding = kernel_size // 2)
        )
        
        self.gp = nn.Sequential(*modules)
        
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
        

class FFA(nn.Module):
    def __init__(self, groups = 3, blocks = 19, channels = 64, kernel_size = 3):
        super(FFA, self).__init__()
        self.gps = groups
        self.channels = channels
        self.kernel_size = kernel_size

        self.preprocess = nn.Conv2d(3, channels, kernel_size, padding = kernel_size // 2)
        self.g1 = Group(channels, kernel_size, blocks = blocks)
        self.g2 = Group(channels, kernel_size, blocks = blocks)
        self.g3 = Group(channels, kernel_size, blocks = blocks)

        self.ca_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels * self.gps, self.channels // 16, 1, padding=0),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.channels // 16, self.channels * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.pa_layer = PALayer(self.channels)

        self.postprocess = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding = kernel_size // 2),
            nn.Conv2d(channels, 3, kernel_size, padding = kernel_size // 2),
        )

    def forward(self, x):
        x1 = self.preprocess(x)
        g1 = self.g1(x1)
        g2 = self.g2(g1)
        g3 = self.g3(g2)

        w = self.ca_layer(torch.cat([g1, g2, g3], dim = 1))
        w = w.view(-1, self.gps, self.channels)[:, :, :, None, None]
        print(w.shape)
        out = w[:, 0, ...] * g1 + w[:, 1, ...] * g2 + w[:, 2, ...] * g3

        out = self.pa_layer(out)
        x1 = self.postprocess(out)

        return x1 + x