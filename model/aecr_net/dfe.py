import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        super(DeformableConvBlock, self).__init__()
        self.stride = stride 
        self.padding = padding 
        self.kernel_size = kernel_size 

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            3 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        off1, off2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([off1, off2], dim=1)
        mask = torch.sigmoid(mask)

        return deform_conv2d(
            x, offset, self.weight, self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            mask=mask
        )

class DFEModule(nn.Module):
    """
    Refactored DFE: Corrected layer naming and removed in-place activations.
    """
    def __init__(self, dim):
        super(DFEModule, self).__init__()
        self.dcn1 = DeformableConvBlock(dim, dim)
        self.relu1 = nn.ReLU(inplace=False)
        self.dcn2 = DeformableConvBlock(dim, dim)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.dcn1(x))
        out = self.relu2(self.dcn2(out))
        return out