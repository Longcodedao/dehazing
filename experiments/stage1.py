import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba


from data.utils import get_haze_transforms, partition_dataset
from torch.utils.data import Subset
from losses import CharbonnierLoss
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

# ==========================================
# 1. HELPER CLASSES & DOWNSAMPLING BLOCKS
# ==========================================

def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise ValueError(f'Pad type [{pad_type}] not recognized')

class AntiAlias_Downsample(nn.Module):
    def __init__(self, channels, pad_type = 'reflect', filt_size = 3, 
                        stride = 2, pad_off = 0):
        super(AntiAlias_Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_type = pad_type

        # Asymmetric padding (round up at the top and round down at the bottom)
        # Perfect when kernel size is 2 
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride 
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels 

        # Define the binomial filter weights
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
            
        # Create a 2D filter by taking the outer product of the 1D filter
        filt = torch.tensor(a[:, None] * a[None, :], dtype = torch.float32)
        filt = filt / torch.sum(filt) # Normalize

        # Reshape to (out_channels, in_channels/groups, kH, kW) for 
        # depthwise convolution
        filt = filt.view(1, 1, filt_size, filt_size)
        filt = filt.repeat(channels, 1, 1, 1)

        # Register as a buffer so PyTorch knows these are NOT trainable parameters
        self.register_buffer('filt', filt)
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride] 
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride] 

        else:
            return F.conv2d(self.pad(inp), self.filt, stride = self.stride, groups = inp.shape[1])

class VariantA_StandardDownsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.down = nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)
    def forward(self, x): return self.down(x)

class VariantB_AntiAliasedDownsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.aa_down = AntiAlias_Downsample(channels=dim_out, filt_size=3, stride=2)
    def forward(self, x): return self.aa_down(self.conv(x))

class BilinearUpsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
    def forward(self, x): return self.conv(self.up(x))


# ==========================================
# 2. FEATURE EXTRACTORS (LOCAL / DILATED / MAMBA)
# ==========================================

class LocalFeatureExtractor(nn.Module):
    def __init__(self, dim, kernel_size=3, expansion_factor=2, dilation=1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        padding = (dilation * (kernel_size - 1)) // 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                      padding=padding, dilation=dilation, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )
    def forward(self, x): return self.net(x)

class PhysBiMambaBlock(nn.Module):
    """ Bidirectional Mamba with spatial smoothing and local gating. """
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_h_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_h_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_v_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_v_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        
        self.fusion_proj = nn.Linear(dim, dim)
        self.spatial_smoothing = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.local_conv = LocalFeatureExtractor(dim, dilation=1)
        self.mixer = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        nn.init.constant_(self.mixer[0].bias, -1.0)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)

        # Horizontal
        out_h_fwd = self.mamba_h_fwd(x_norm)
        out_h_bwd = torch.flip(self.mamba_h_bwd(torch.flip(x_norm, dims=[1])), dims=[1])

        # Vertical
        x_v_flat = x_norm.view(B, H, W, C).permute(0, 2, 1, 3).flatten(1, 2)
        out_v_fwd = self.mamba_v_fwd(x_v_flat)
        out_v_bwd = torch.flip(self.mamba_v_bwd(torch.flip(x_v_flat, dims=[1])), dims=[1])
        out_v_fwd = out_v_fwd.view(B, W, H, C).permute(0, 2, 1, 3).flatten(1, 2)
        out_v_bwd = out_v_bwd.view(B, W, H, C).permute(0, 2, 1, 3).flatten(1, 2)

        global_feat = self.fusion_proj(out_h_fwd + out_h_bwd + out_v_fwd + out_v_bwd)
        global_feat = self.spatial_smoothing(global_feat.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        
        local_feat = self.local_conv(x).flatten(2).transpose(1, 2)
        z = self.mixer(torch.cat([global_feat, local_feat], dim=-1))
        fused = self.out_proj(global_feat * z + local_feat * (1 - z))
        
        return residual + self.dropout(fused.transpose(1, 2).view(B, C, H, W))


# ==========================================
# 3. THE ABLATION MODEL (MASTER CLASS)
# ==========================================

class AblationPhysicsEstimator(nn.Module):
    """
    Variants Summary:
    A: Standard CNN (Stride-2 Conv Downsampling)
    B: Physical Mamba + Anti-Aliased Downsampling
    C: Anti-Aliased CNN (No Mamba, isolate BlurPool effect)
    D: Dilated AA-CNN (Anti-Aliased Downsampling + Dilated Bottleneck)
    """
    def __init__(self, variant='D', in_channels=3, base_dim=32):
        super().__init__()
        self.variant = variant
        
        # Initial Encoder
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.enc1 = nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1)

        # Logic for Downsampling and Bottleneck
        if variant == 'A':
            self.down1 = VariantA_StandardDownsample(base_dim, base_dim * 2)
            self.down2 = VariantA_StandardDownsample(base_dim * 2, base_dim * 4)
            self.bottleneck = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, padding=1)
        else:
            # Variants B, C, D all use Anti-Aliased Downsampling
            self.down1 = VariantB_AntiAliasedDownsample(base_dim, base_dim * 2)
            self.down2 = VariantB_AntiAliasedDownsample(base_dim * 2, base_dim * 4)
            
            if variant == 'B':
                self.bottleneck = PhysBiMambaBlock(dim=base_dim * 4)
            elif variant == 'C':
                self.bottleneck = nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, padding=1)
            elif variant == 'D':
                # Use Dilation=2 for larger receptive field without sequential artifacts
                self.bottleneck = LocalFeatureExtractor(base_dim * 4, dilation=2)
            else:
                raise ValueError("Variant must be A, B, C, or D")

        # Atmospheric Light (A) Head
        self.A_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base_dim * 4, 3), nn.Sigmoid() 
        )
        
        # Decoder with Skip Connections
        self.up1 = BilinearUpsample(base_dim * 4, base_dim * 2)
        self.dec1_conv = nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=1) 
        self.dec1 = nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1)
        
        self.up2 = BilinearUpsample(base_dim * 2, base_dim)
        self.dec2_conv = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec2 = nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1)
        
        # Transmission (t) Head
        self.t_head = nn.Sequential(nn.Conv2d(base_dim, 1, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(self.init_conv(x))
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        b = self.bottleneck(d2)
        A = self.A_head(b).view(-1, 3, 1, 1)
        
        u1 = torch.cat([self.up1(b), e2], dim=1)
        u1 = self.dec1(self.dec1_conv(u1))
        
        u2 = torch.cat([self.up2(u1), e1], dim=1)
        u2 = self.dec2(self.dec2_conv(u2))
        
        return self.t_head(u2), A


class RESIDE_Indoor(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.root_dir = Path(dataset_path)
        self.metadata_csv = pd.read_csv(self.root_dir / "metadata.csv")

        self.transform = transform
        self.data = []
        
        for idx, row in self.metadata_csv.iterrows():
            clean_path = self.root_dir / row["clear_image_path"]
            hazy_paths_str = row["hazy_image_paths"]
            hazy_image_paths = [
                path.strip()
                for path in hazy_paths_str.strip("[]").replace("'", "").split(",")
            ]
            list_hazy_paths = [
                self.root_dir / hazy_path for hazy_path in hazy_image_paths
            ]
            
            for hazy_path in list_hazy_paths:
                # --- NEW LOGIC: Deduce the Transmission Map Path ---
                # Example: hazy_path.name is "1_1_0.90179.png"
                hazy_filename = hazy_path.name
                parts = hazy_filename.split('_')
                
                # Reconstruct trans filename: "1_1.png"
                if len(parts) >= 2:
                    trans_filename = f"{parts[0]}_{parts[1]}.png"
                else:
                    trans_filename = hazy_filename # Fallback just in case
                
                trans_path = self.root_dir / "trans" / trans_filename
                # ---------------------------------------------------

                data_item = {
                    "index": idx, 
                    "clean": clean_path, 
                    "hazy": hazy_path,
                    "trans": trans_path # Store the trans path
                }

                self.data.append(data_item)

    def __repr__(self):
        return "RESIDE Indoor"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]
        trans_path = data_item["trans"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
            # Load transmission map as Grayscale ("L")
            trans_img = Image.open(trans_path).convert("L") 
        except FileNotFoundError:
            print(f"Error: Missing image file at {clean_path}, {hazy_path}, or {trans_path}. Skipping")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            # IMPORTANT WARNING: 
            # If your 'get_haze_transforms' function only expects 2 inputs, 
            # you must update it to accept and return 3 inputs!
            clean_img, hazy_img, trans_img = self.transform(clean_img, hazy_img, trans_img)
        else:
            # Fallback tensorization
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )
            # Add channel dimension to grayscale image (H, W) -> (1, H, W)
            trans_img = (
                torch.as_tensor(np.array(trans_img)).unsqueeze(0).float() / 255.0
            )

        return hazy_img, clean_img, trans_img