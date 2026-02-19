import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def get_dark_channel(img, kernel_size = 15):
    """
    Calculates the Dark Channel Prior (DCP)
    img: Tensor of shape (B, 3, H, W)
    """
    # Step 1: Minimum across color channels
    min_channel, _ = torch.min(img, dim = 1, keepdim = True)

    # Step 2: minimum over a local patch Omega(x)
    pad = kernel_size // 2
    dark_channel = F.max_pool2d(-min_channel, kernel_size, stride = 1, padding = pad)

    return -dark_channel

"""
Dehamer Position Embedding:
Create the 3D position Embedding from the Dark Channel Prior 
From the paper Dehamer since only the spatial position order 
is not enough for the variational haze densities of different spatial 
regions in a hazy image
"""
class DeHamerPosEmbed(nn.Module):
    def __init__(self, channels = 96, p_total = 32, temperature = 10000, 
                  scale = 2 * math.pi):
        super().__init__()
        self.channels = channels
        self.p_total = p_total
        self.temperature = temperature
        self.scale = scale 

    def forward(self, x, haze_density):
        """
        x: Input feature tokens (B, C, H, W)
        haze_density: DCP map (B, 1, H, W)
        """
        B, C, H, W = x.shape
        B_D, C_D, H_D, W_D = haze_density.shape
        assert B_D == B and C_D == 1 and H == H_D and W == W_D

        device = x.device
        eps = 1e-6

        y_embed = torch.arange(H, device = device).view(H, 1).repeat(1, W).float()
        x_embed = torch.arange(W, device = device).view(1, W).repeat(H, 1).float()