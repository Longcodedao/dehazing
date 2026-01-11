import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from yacs.config import CfgNode as CN
import os 
from mamba_ssm import Mamba

# Import the defaults we just created
from .config_model import get_model_cfg_defaults


# --- HELPER: Config Loader ---
def get_model_config(config_path_or_name):
    """
    Returns a YACS CfgNode.
    Args:
        config_path_or_name: "small", "large", or path to .yaml file
    """
    cfg = get_model_cfg_defaults()
    
    # 1. Handle Shortnames
    if config_path_or_name == "small":
        config_path_or_name = "configs/model_cfgs/small.yaml"
    elif config_path_or_name == "large":
        config_path_or_name = "configs/model_cfgs/large.yaml"
        
    # 2. Merge YAML if exists
    if os.path.exists(config_path_or_name):
        cfg.merge_from_file(config_path_or_name)
        cfg.freeze() # Prevent accidental modification
        return cfg
    else:
        # Fallback or Error
        print(f"Warning: Config {config_path_or_name} not found. Using Defaults.")
        return cfg

        

class PhysConvNeXtBlock(nn.Module):
    """
    ConvNeXt V2 Block: Best for local textures and edges.
    Includes Adaptive Layer Norm (AdaLN) for Time Embedding injection.
    """
    def __init__(self, dim, mult=2):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x, t_emb=None):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        # Adaptive Layer Norm (Time Injection)
        if t_emb is not None:
            # t_emb is (N, C) -> Scale & Shift
            x = self.norm(x)
            scale, shift = t_emb.chunk(2, dim=1)
            x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        else:
            x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return input + x
        

class PhysBiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block (BiMamba)
    Scans the image Forward AND Backward so the top-left pixel
    can 'see' the bottom-right pixel.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.mamba_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

        # 2. THE FUSION LAYER (The upgrade)
        # Takes both directions (dim * 2) and learns how to combine them back to (dim)
        self.fusion_linear = nn.Linear(dim * 2, dim)
        
        # Optional: A Gate to let the network choose emphasis
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    def forward(self, x, t_emb=None):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        # 1. Prepare Sequence: (B, C, H, W) -> (B, L, C)
        x_flat = x.flatten(2).transpose(1, 2) # (B, L, C)
        x_norm = self.norm(x_flat)

        # 2. Inject Time
        if t_emb is not None:
             # Take only the first half (scale) for simple addition
            t_val, _ = t_emb.chunk(2, dim=1)
            # print(f'x_norm: {x_norm.shape}')
            # print(f't_val: {t_val.unsqueeze(1).shape}')
            x_norm = x_norm + t_val.unsqueeze(1)

        # 3. Bidirectional Scanning
        
        # --- Forward Scan (Standard) ---
        out_fwd = self.mamba_fwd(x_norm)
        
        # --- Backward Scan (Flip -> Scan -> Flip Back) ---
        x_flip = torch.flip(x_norm, dims=[1]) # Reverse the sequence
        out_bwd = self.mamba_bwd(x_flip)
        out_bwd = torch.flip(out_bwd, dims=[1]) # Reverse back to original order
        
        # 4. Combine
        # --- Learned Fusion (Better than Averaging) ---
        
        # Concatenate features: Shape becomes (B, L, 2*C)
        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        
        # Calculate a Gate (0 to 1) deciding flow importance
        # "z" tells us how much to listen to the mixture
        z = self.fusion_gate(combined)
        
        # Project back to original dimension
        x_fused = self.fusion_linear(combined)
        
        # Gated Activation: This is very stable for Mamba
        x_out = x_fused * z
        
        # 5. Reshape back to Image
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        return residual + x_out


class FM_PhysMamba_UNET(nn.Module):
    def __init__(self, model_cfg_path="small", in_channels=3):
        """
        Args:
            model_cfg_path: "small", "large", or path to a .yaml file
        """
        super().__init__()
        
        # 1. Load Config into CN
        if isinstance(model_cfg_path, CN):
            self.cfg = model_cfg_path # Already loaded
        else:
            self.cfg = get_model_config(model_cfg_path)

        # 2. Extract Params from CN
        base_dim = self.cfg.BASE_DIM
        dim_mults = self.cfg.DIM_MULTS
        self.physics_guided = self.cfg.PHYSICS_GUIDED
        enc_blocks_list = self.cfg.ENCODER_BLOCKS
        dec_blocks_list = self.cfg.DECODER_BLOCKS
        
        self.dims = [base_dim * m for m in dim_mults]
        
        # --- Time Embedding ---
        time_dim = base_dim * self.cfg.TIME_DIM_MULT
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_time_projs = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()

        # --- ENCODER ---i
        self.init_conv = nn.Conv2d(in_channels, self.dims[0], 3, 1, 1)
        
        self.downs = nn.ModuleList()       # Processing Blocks
        self.downsamples = nn.ModuleList() # Downsampling Layers (Separated)
        
        for i in range(len(self.dims) - 1):
            dim_in, dim_out = self.dims[i], self.dims[i+1]
            self.down_time_projs.append(nn.Linear(time_dim, dim_in * 2))

            # Build Processing Stack
            blocks = []
            blocks.append(PhysConvNeXtBlock(dim_in)) # Always start with Conv
            
            # Add specified number of Mamba blocks
            num_mamba = enc_blocks_list[i] if i < len(enc_blocks_list) else 1
            for _ in range(num_mamba):
                blocks.append(PhysBiMambaBlock(dim_in))
            
            self.downs.append(nn.Sequential(*blocks))

            # Separate Downsampling Layer
            self.downsamples.append(nn.Conv2d(dim_in, dim_out, 4, 2, 1)) 

        # --- BOTTLENECK ---
        mid_dim = self.dims[-1]
        self.mid_time_proj = nn.Linear(time_dim, mid_dim * 2)
        
        self.mid_block1 = PhysBiMambaBlock(mid_dim)
        self.mid_block2 = PhysBiMambaBlock(mid_dim)
        
        # Physics Head A
        self.atm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid_dim, 64), nn.SiLU(),
            nn.Linear(64, 3), nn.Sigmoid() 
        )

        # --- DECODER ---
        self.ups = nn.ModuleList()
        # Iterate backwards
        for idx, i in enumerate(range(len(self.dims)-2, -1, -1)):
            dim_in, dim_out = self.dims[i+1], self.dims[i]
            self.up_time_projs.append(nn.Linear(time_dim, dim_out * 2))
            
            layers = []
            layers.append(nn.ConvTranspose2d(dim_in, dim_out, 2, 2)) # Upsample
            layers.append(nn.Conv2d(dim_out*2, dim_out, 1)) # Reduce concatenated channels
            
            # Stack Blocks
            num_mamba = dec_blocks_list[i] if i < len(dec_blocks_list) else 1
            for _ in range(num_mamba):
                if i > 0: # Deeper layers = Mamba
                    layers.append(PhysBiMambaBlock(dim_out))
                else: # Shallow layers = Conv
                    layers.append(PhysConvNeXtBlock(dim_out))
            
            layers.append(PhysConvNeXtBlock(dim_out)) 
            self.ups.append(nn.Sequential(*layers))

        # Physics Head T
        self.trans_head = nn.Sequential(
            nn.Conv2d(self.dims[0], 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid() 
        )
        
        self.final_conv = nn.Conv2d(self.dims[0], 3, 1)

    def get_sinusoidal_emb(self, t, device):
        half_dim = self.dims[0] // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t):
        t_emb_raw = self.get_sinusoidal_emb(t, x.device) 
        t_vec = self.time_mlp(t_emb_raw)
        
        h = self.init_conv(x)
        skips = [] # Initialize empty list, NOT [h]
        
        # --- ENCODER ---
        # Zip allows us to iterate Processing and Downsampling in sync
        for i, (block_stack, down_layer) in enumerate(zip(self.downs, self.downsamples)):
            t_emb = self.down_time_projs[i](t_vec)
            
            # 1. Process Features
            for layer in block_stack:
                if isinstance(layer, (PhysConvNeXtBlock, PhysBiMambaBlock)):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            
            # 2. SAVE Skip Connection (Before Downsampling!)
            skips.append(h)
            
            # 3. Downsample
            h = down_layer(h)
            
        # --- BOTTLENECK ---
        t_emb_mid = self.mid_time_proj(t_vec)
        h = self.mid_block1(h, t_emb_mid)
        h = self.mid_block2(h, t_emb_mid)
        
        A_pred = self.atm_head(h).view(-1, 3, 1, 1)
        
        # --- DECODER ---
        for i, block_stack in enumerate(self.ups):
            # 1. Upsample
            h = block_stack[0](h) 
            
            # 2. Retrieve Skip Connection
            if len(skips) > 0:
                skip = skips.pop()
                # Concatenate (skip is High Res, h is High Res)
                h = torch.cat([h, skip], dim=1)
            else:
                # Fallback if dimensions don't align perfectly (shouldn't happen with correct config)
                pass

            # 3. Reduce Channels
            h = block_stack[1](h) 
            
            t_emb = self.up_time_projs[i](t_vec)
            
            # 4. Process Decoder Blocks
            for layer in block_stack[2:]:
                if isinstance(layer, (PhysConvNeXtBlock, PhysBiMambaBlock)):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
                    
        t_map = self.trans_head(h)
        if self.physics_guided:
            h = h * (1 + t_map)
        v_pred = self.final_conv(h)
        
        return v_pred, t_map, A_pred