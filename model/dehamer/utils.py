import torch
import torch.nn as nn
import torch.nn.functional as F

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

        y_embed = y_embed / (H + eps) * self.scale
        x_embed = x_embed / (W + eps) * self.scale

        # Haze density normalization
        z_embed = haze_density.squeeze(1).float() # B, H, W
        z_max, _ = z_embed.view(B, -1).max(dim = 1)
        z_max = z_max.view(B, 1, 1)
        z_embed = z_embed / (z_max + eps) * self.scale

        # Temperature-Based Frequencey Scaling 
        # We calculate dimensions for the encoding
        dim_t = torch.arange(self.p_total, dtype = torch.float32, device = device)
        # 1000 ** (2i / p_total) 
        # We divide the dim_t by 2 because 2 pairs of odd and even indices 
        # corresponds to sine and cosine 
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.p_total)

            
        def encode(pos_tensor, is_spatial = True):
            # pos_tensor: (H, W) for spatial or (B, H, W) for haze
            if is_spatial: 
                pos_tensor = pos_tensor.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1)
            else:
                pos_tensor = pos_tensor.unsqueeze(-1)

            # Apply Frequency scaling
            # The division by dim_t follows the author's logic: pos / (temp^(2i/d))
            encoded = pos_tensor / dim_t

            pos_sin = encoded[:, :, :, 0::2].sin()
            pos_cos = encoded[:, :, :, 1::2].cos()

            # Concatenate along the feature dimension to get p_total (32)
            return torch.stack((pos_sin, pos_cos), dim = 4).flatten(3) # (B, H, W, 32)

        # (B, 32, H, W)
        pe_x = encode(x_embed).permute(0, 3, 1, 2)
        pe_y = encode(y_embed).permute(0, 3, 1, 2)
        pe_z = encode(z_embed, is_spatial = False).permute(0, 3, 1, 2)

        # Final 3D concatenation
        pe_3d = torch.cat((pe_x, pe_y, pe_z), dim = 1) # (B, 96, H, W)

        return x + pe_3d
