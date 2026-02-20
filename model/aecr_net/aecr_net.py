import torch
import torch.nn as nn 
import torch.nn.functional as F
from .dfe import DeformableConvBlock, DFEModule

## Pixel Attention Block 
class PALayer(nn.Module):
    def __init__(self, channel, reduce_ratio = 8):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduce_ratio, 1, padding = 0, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel // reduce_ratio, 1, 1, padding = 0, bias = True),
            nn.Sigmoid()
        )

    def forward(self, x):
        atten = self.pa(x)
        return atten * x

## Channel Attention Block 
class CALayer(nn.Module):
    def __init__(self, channel, reduce_ratio = 8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduce_ratio, 1, padding = 0, bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel // reduce_ratio, channel, 1, padding = 0, bias = True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(x)
        return x * y
        
"""
FABlock for mixing both global information (haze environment) and the local features
"""
class FABlock(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super(FABlock, self).__init__()
        self.padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding = self.padding)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding = self.padding)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(x)
        res = self.conv2(x)
        res = self.calayer(x)
        res = self.palayer(x)

        return res


"""
Adaptive Mixup BLock
This block is for mixing up features (down-sampling and up-sampling)
"""
class AdaptiveMixup(nn.Module):
    def __init__(self, init_value = -0.80):
        super(AdaptiveMixup, self).__init__()
        # Learnable factor theta for sigmoid fusion 
        self.theta = nn.Parameter(torch.tensor(init_value), requires_grad = True)

    def forward(self, down_feat, up_feat):
        # mix_factor = self.mix_block(self.theta)
        mix_factor = torch.sigmoid(self.theta)
        out = mix_factor.expand_as(down_feat) * down_feat + \
                (1 - mix_factor.expand_as(up_feat)) * up_feat

        return out 


class AECRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super(AECRNet, self).__init__()
        
        # 4x Downsampling 
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, dim, 3, 1, 1), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(True))

        # 6 FA Blocks 
        self.fa_blocks = nn.Sequential(*[FABlock(dim) for _ in range(6)])
        
        # NOTE: DFE module typically uses Deformable Convolutions [cite: 510]
        # For standard environments, we use a placeholder Conv here.
        self.dfe = DFEModule(dim)

        # 4x Upsampling [cite: 496]
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(True))
        self.up3 = nn.Conv2d(dim, out_channels, 3, 1, 1)

        # Adaptive Mixup Connections [cite: 498]
        self.mix1 = AdaptiveMixup()
        self.mix2 = AdaptiveMixup()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        out = self.fa_blocks(d3)
        out = self.dfe(out)

        u1 = self.up1(out)
        u1 = self.mix1(d2, u1) # Mix with 2nd down layer [cite: 503]
        
        u2 = self.up2(u1)
        u2 = self.mix2(d1, u2) # Mix with 1st down layer [cite: 503]
        
        return self.up3(u2)