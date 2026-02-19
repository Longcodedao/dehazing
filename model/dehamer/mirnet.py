import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .antialias import AntiAlias_Downsample


"""
Residual Resizing Modules (RRM):
Combining Residual Connections for tackling the gradient vanishing problem
While applying anti-aliasing modules before downsampling for protecting 
information
"""
# Using the convolution 2D with the overlay
def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size//2), 
        bias=bias, stride = stride)

# Residual DownSample
# Uses the residual gradient that makes deep networks easy to train
# But applies the lowpass filtering to preserve the anti-aliasing operation 
# which preserves the network's shift-equivariance
class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride = 1, padding = 0, bias = bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1, bias = bias),
            AntiAlias_Downsample(in_channels, filt_size = 3, stride = 2),
            nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = bias)
        )

        self.bot = nn.Sequential(
            AntiAlias_Downsample(in_channels, filt_size = 3, stride = 2),
            nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = bias)
        )

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot 
        
        return out


# Residual UpSample 
class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride = 1, padding = 0, bias = bias),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = bias)
        )

        self.bot = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = bias),
            nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0, bias = bias)
        )

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot 
        
        return out


# Downsample 
class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride = 2):
        super(DownSample, self).__init__()
        num_blocks = int(round(math.log(scale_factor, stride)))

        if (stride ** num_blocks) != scale_factor:
            raise ValueError(f"scale_factor ({scale_factor}) must be a perfect power of stride ({stride}).") 

        modules_body = []
        for i in range(num_blocks):
            out_channels = int(in_channels * stride)
            modules_body.append(
                ResidualDownSample(in_channels, out_channels)
            )

            in_channels = out_channels 

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class UpSample(nn.Module):
    def __init__(UpSample, in_channels, scale_factor, stride = 2, kernel_size = 3):
        super(UpSample, self).__init__()
        num_blocks = int(round(math.log(scale_factor, stride)))

        if (stride ** num_blocks) != scale_factor:
            raise ValueError(f"scale_factor ({scale_factor}) must be a perfect power of stride ({stride}).") 

        modules_body = []
        for i in range(num_blocks):
            out_channels = int(in_channels //  stride)
            modules_body.append(
                ResidualUpSample(in_channels, out_channels)
            )

            in_channels = out_channels 

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


"""
Multi-scale Residual Block (MRB):
Capable of capturing the multi-resolutions output at the same time.
Getting the sptaially precise high resolution and rich contextual information in low resolution
"""
# DAU: Dual Attention Unit: Getting both Channel and Spatial Information
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7, bias = False):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * y



class DAU(nn.Module):
    """Dual Attention Unit"""
    def __init__(self, channels, bias = False):
        super(DAU, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        res = x
        y = self.prelu(self.conv1(x))
        y = self.conv2(y)
        
        # Apply Channel and Spatial Attention
        c_out = self.ca(y)
        s_out = self.sa(y)
        
        # Summing parallel attention branches 
        return res + c_out + s_out


# SKFF: Merge multi-scale features together by weighting for the scales 
# of feature maps for each resolution level
class SKFF(nn.Module):
    def __init__(self, in_channels, height = 3, reduction = 8, bias = False):
        super(SKFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.height = height
        
        # 'r' channel downscaling ratio
        d = max(int(in_channels / reduction), 4)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=False),
            nn.PReLU()
        )

        # Parallel channel upscaling layers for the 3 streams
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(
                nn.Conv2d(d, in_channels, kernel_size = 1, stride = 1, bias = bias)
            )

    def forward(self, inp_feats):
        B, C, _, _ = inp_feats[0].shape

        # B, 3, C, H, W
        inp_feats = torch.stack(inp_feats, dim = 1)
        
        L = torch.sum(inp_feats, dim = 1)
        s = self.avg_pool(L)
        z = self.conv_du(s)

        attention_vectors = [fc(z) for fc in self.fcs]
        attention_vectors = torch.stack(attention_vectors, dim = 1)
        attention_vectors = F.softmax(attention_vectors, dim = 1)

        # Aggregation
        U = (inp_feats * attention_vectors).sum(dim = 1)
        return U


class MSRB(nn.Module):
    def __init__(self, in_channels, stride, height = 3, width = 2, bias = False):
        """
        Args:
            in_channels: The number of channels entering the block.
            stride:  The downsampling factor between the multi-resolution stream
            height: The number of parallel multi-resolution streams (default 3 in MIRNet).
            width: The number of sequential DAU blocks per stream
            reduction: The reduction ratio used inside the SKFF module.
        """
        super(MSRB, self).__init__()

        self.in_channels = in_channels 
        self.stride = stride
        self.height =  height
        self.width = width

        self.daus = nn.ModuleList()
        for i in range(height):
            stream_channels = int(in_channels * (stride ** i))
            dau_blocks = [DAU(stream_channels, bias = bias) for _ in range(width)]
            self.daus.append(nn.ModuleList(dau_blocks))

        self.routing = nn.ModuleDict()
        for source in range(height):
            for target in range(height):
                if source == target:
                    continue

                source_channels = int(in_channels * (stride ** source))
                scale_factor = stride ** abs(target - source)

                if source < target:
                    # print(f'[Down] {source}_{target}: source_channel = {source_channels}, scale_factor = {scale_factor}, stride = {stride}')
                    self.routing[f'{source}_{target}'] = DownSample(source_channels, scale_factor, stride)
                else:
                    # print(f'[Up] {source}_{target}: source_channel = {source_channels}, scale_factor = {scale_factor}, stride = {stride}')

                    self.routing[f'{source}_{target}'] = UpSample(source_channels, scale_factor, stride)

        self.last_upsamples = nn.ModuleList()
        for i in range(1, height):
            stream_channels = int(in_channels * (stride ** i))
            scale_factor = stride ** i
            # print(f'[Last-Up] {i}: source_channel = {stream_channels}, scale_factor = {scale_factor}, stride = {stride}')
            self.last_upsamples.append(UpSample(stream_channels, scale_factor, stride))
        
        self.skffs = nn.ModuleList([
            SKFF(int(in_channels * (stride ** i)), height=height, bias=bias) 
            for i in range(height)
        ])
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1, bias = bias)

    def forward(self, x):
        # --- STAGE 1: Initialization (First Column of DAUs) ---
        streams = []
        current_feat = x 
        
        for j in range(self.height):
            if j == 0:
                current_feat = self.daus[j][0](current_feat)
            else:
                current_feat = self.routing[f'{j - 1}_{j}'](current_feat)
                # print(current_feat.shape)

                current_feat = self.daus[j][0](current_feat)
            streams.append(current_feat)

        # --- STAGE 2: Multi-Scale Mesh Routing ---
        for w in range(1, self.width):
            next_streams = []

            for target in range(self.height):
                aligned_feats = []
                for source in range(self.height):
                    # print(f'{source}_{target}')
                    current_stream = streams[source]
                    if source == target:
                        aligned_feats.append(current_stream)
                    else:
                        aligned_feats.append(
                            self.routing[f'{source}_{target}'](current_stream)
                        )

                # Fused the align feaures with SKFF
                # print(f"Aligned Features has {len(aligned_feats)} with shape [{aligned_feats[0].shape}]")
                fused = self.skffs[target](aligned_feats)

                # Pass the fused feature through the next DAU in this stream
                next_streams.append(
                    self.daus[target][w](fused)
                )
            streams = next_streams

        # --- STAGE 3: Final Aggregation
        final_aligned = [streams[0]]
        for source in range(1, self.height):
            up_sample = self.last_upsamples[source - 1](streams[source])
            # print(f"Upsample at {source} has shape: {up_sample.shape}")
            final_aligned.append(up_sample)

        out = self.skffs[0](final_aligned)
        out = self.conv_out(out)

        return out + x

"""
Recursive Residual Group (RRG)
Group all of the blocks MSRB together but has the residual connection in the end
""" 
class RRG(nn.Module):
    def __init__(self, in_channels, num_blocks, height, width, stride, bias = False):
        super(RRG, self).__init__()
        blocks = [
            MSRB(in_channels, stride, height = height, width = width, bias = bias) for _ in range(num_blocks)
        ]
        blocks.apped(conv(in_channels, in_channels, kernel_size = 3))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


"""
Final MIRNet: Combinations of RRG blocks together
"""
class MIRNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, n_feat = 64,
                kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, 
                width=2, bias=False):
        super(MIRNet, self).__init__()
        
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size,
                                 padding=(kernel_size - 1) // 2, bias=bias)

        modules_body = [RRG(n_feat, n_MSRB, height, width, stride, bias) for _ in range(n_RRG)]
        self.body = nn.Sequential(*modules_body)

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.body(h)
        h = self.conv_out(h)
        h += x
        return h