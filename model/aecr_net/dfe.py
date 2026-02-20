import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

# Deformable Convolution 2D 
# Deformable Convolutional Networks paper by (Jifeng Dai et al.)
# https://arxiv.org/abs/1703.06211
class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, 
                 stride = 1, padding = 1, bias = False):
        super(DeformableConvBlock, self).__init__()
        self.stride = stride 
        self.padding = padding 
        self.kernel_size = kernel_size 

        # Weight for the actual deformable convolution
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        ## Conv to generate offsets (2 * k*k) and mask (1 * k*k)
        # For 3x3 kernel, this is 18 + 9 = 27 channels
        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            3 * kernel_size * kernel_size, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding, 
            bias = True
        )

        self.init_weights()

    def init_weights(self):
        # Standard Kaiming init for weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Intialize offset/mask conv to zero
        # This ensure the module starts as a standard convolution
        # Maybe start asking questions: This behaviour is more stable though I guess
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)
        

    def forward(self, x):
        # Generate offsets and mask 
        out = self.conv_offset_mask(x)
        off1, off2, mask = torch.chunk(out, 3, dim = 1)

        # Concatenate all offsets together
        offset = torch.cat([off1, off2], dim = 1)
        # Mask must be between 0 and 1 
        mask = torch.sigmoid(mask)

        return deform_conv2d(
            x, offset, self.weight, self.bias,
            stride = (self.stride, self.stride),
            padding = (self.padding, self.padding),
            mask = mask
        )


class DFEModule(nn.Module):
    """
    Dynamic Feature Enhancement (DFE) Module for AECR-Net.
    Consists of two deformable convolutional layers to expand the 
    receptive field and capture spatially structured information.
    """
    def __init__(self, dim):
        super(DFEModule, self).__init__()
        self.dcn1 = DeformableConvBlock(dim, dim, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU(inplace = True)

        self.dcn1 = DeformableConvBlock(dim, dim, kernel_size = 3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU(inplace = True)


    def forward(self, x):
        out = self.relu1(self.dcn1(x))
        out = self.relu1(self.dcn1(out))

        return out