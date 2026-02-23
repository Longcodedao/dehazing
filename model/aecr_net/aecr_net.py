import torch
import torch.nn as nn 
import torch.nn.functional as F
from .dfe import DFEModule

class PALayer(nn.Module):
    def __init__(self, channel, reduce_ratio=8):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduce_ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduce_ratio, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pa(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduce_ratio=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduce_ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduce_ratio, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class FABlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(FABlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=padding)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        # Corrected: Chain the results properly
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        return res + x # Residual connection

class AdaptiveMixup(nn.Module):
    def __init__(self, init_value=-0.80):
        super(AdaptiveMixup, self).__init__()
        self.theta = nn.Parameter(torch.tensor(init_value), requires_grad=True)

    def forward(self, down_feat, up_feat):
        mix_factor = torch.sigmoid(self.theta)
        # Using expanded factor for clarity and backprop stability
        return mix_factor * down_feat + (1 - mix_factor) * up_feat

class AECRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super(AECRNet, self).__init__()
        
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, dim, 3, 1, 1), nn.ReLU(False))
        self.down2 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(False))
        self.down3 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(False))

        # BottleNeck
        self.fa_blocks = nn.Sequential(*[FABlock(dim) for _ in range(6)])
        self.dfe = DFEModule(dim)

        # Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(False))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(False))
        self.up3 = nn.Conv2d(dim, out_channels, 3, 1, 1)

        # Skip Connections
        self.mix1 = AdaptiveMixup()
        self.mix2 = AdaptiveMixup()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Features
        out = self.fa_blocks(d3)
        out = self.dfe(out)

        # Decoder with Skip Connections
        u1 = self.up1(out)
        u1 = self.mix1(d2, u1)
        
        u2 = self.up2(u1)
        u2 = self.mix2(d1, u2)
        
        return self.up3(u2)