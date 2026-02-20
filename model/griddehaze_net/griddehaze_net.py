import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self, in_channels, channel_factor = 2, kernel_size = 3):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * channel_factor, 
            kernel_size, 
            stride = 2,
            padding = kernel_size // 2   
        )

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, channel_factor = 2, kernel_size = 3):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            in_channels // channel_factor,
            kernel_size, 
            padding = kernel_size // 2
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2, mode = "bilinear", align_corners = True)
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding = 1)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding = 1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual 

        return self.relu(out)


"""
CBAM Components block 
"""
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels // reduction, channels, 1, bias = False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        out = torch.cat([avg_out, max_out], dim = 1)
        out = self.conv(out)

        return self.sigmoid(out) * x

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reductions = 16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reductions)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class GridBlock(nn.Module):
    def __init__(self, channels):
        super(GridBlock, self).__init__()
        self.res_block = ResidualBlock(channels)
        self.attention = CBAM(channels)

    def forward(self, x):
        x = self.res_block(x)
        x = self.attention(x)

        return x

# GridDehazeNet 
class GridDehazeNet(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 16, num_grid_blocks = 4):
        super(GridDehazeNet, self).__init__()

        # Preprocessing module
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Encoder (3 levels)
        self.down1 = DownSample(base_channels, channel_factor = 2)   # 32 channels
        self.down2 = DownSample(base_channels * 2, channel_factor = 2)  # 64 channels
        self.down3 = DownSample(base_channels * 4, channel_factor = 2)  # 128 channels

        # Grid blocks at each level
        # Level 0 (original resolution)
        self.grid_blocks_0 = nn.ModuleList([
            GridBlock(base_channels) for _ in range(num_grid_blocks)
        ])

        # Level 1 
        self.grid_blocks_1 = nn.ModuleList([
            GridBlock(base_channels * 2) for _ in range(num_grid_blocks)
        ])

        # Level 2
        self.grid_blocks_2 = nn.ModuleList([
            GridBlock(base_channels * 4) for _ in range(num_grid_blocks)
        ])

        # Level 3
        self.grid_blocks_3 = nn.ModuleList([
            GridBlock(base_channels * 8) for _ in range(num_grid_blocks)
        ])


        # Decoder 
        self.up3 = UpSample(base_channels * 8, channel_factor = 2)  # -> 64 channels
        self.up2 = UpSample(base_channels * 4, channel_factor = 2)  # -> 32 channels
        self.up1 = UpSample(base_channels * 2, channel_factor = 2)  # -> 16 channels

        # Post-processing
        self.post_process = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        x0 = self.pre_process(x)

        # Level 0 
        features_0 = x0
        for grid_block in self.grid_blocks_0:
            features_0 = grid_block(features_0)

        # Level 1
        x1 = self.down1(features_0)
        features_1 = x1
        for grid_block in self.grid_blocks_1:
            features_1 = grid_block(features_1)

        # Level 2
        x2 = self.down2(features_1)
        features_2 = x2
        for grid_block in self.grid_blocks_2:
            features_2 = grid_block(features_2)

        # Level 3
        x3 = self.down3(features_2)
        features_3 = x3
        for grid_block in self.grid_blocks_3:
            features_3 = grid_block(features_3)

        # Decoder path with skip connections
        up3 = self.up3(features_3)
        up3 = up3 + features_2

        up2 = self.up2(up3)
        up2 = up2 + features_1

        up1 = self.up1(up2)
        up1 = up1 + features_0

        # Post-process
        output = self.post_process(up1)

        # Residual connection with input
        output = output + x

        return output