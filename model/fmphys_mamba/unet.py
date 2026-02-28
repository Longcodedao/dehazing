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
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        
        if t_emb is not None:
            x = self.norm(x)
            scale, shift = t_emb.chunk(2, dim=1)
            x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        else:
            x = self.norm(x)
            
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return inp + x


class LocalFeatureExtractor(nn.Module):
    """ 
    Adaptive Parallel Branch.
    Uses Inverted Bottleneck (Expand -> Depthwise -> Project).
    Automatically calculates padding to keep spatial dimensions constant.
    """
    def __init__(self, dim, kernel_size=3, expansion_factor=2, dilation=2):
        super().__init__()
        
        hidden_dim = int(dim * expansion_factor)
        
        # Dynamic Padding Calculation:
        # P = (dilation * (kernel_size - 1)) / 2
        # This ensures the output size equals the input size.
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.net = nn.Sequential(
            # 1. Pointwise Expansion
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            
            # 2. Adaptive Depthwise Conv
            nn.Conv2d(hidden_dim, hidden_dim, 
                      kernel_size=kernel_size, 
                      padding=padding, 
                      dilation=dilation,
                      groups=hidden_dim), # Depthwise
            nn.GELU(),
            
            # 3. Pointwise Projection
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)
        

class PhysBiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block (BiMamba)
    Scans the image Forward AND Backward so the top-left pixel
    can 'see' the bottom-right pixel.
    """
    def __init__(self, dim, dropout = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # --- Horizontal Mamba -----
        self.mamba_h_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_h_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

        # --- Vertical Mamba ---
        self.mamba_v_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_v_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        
        # Fuses Fwd+Bwd direction
        self.fusion_linear = nn.Linear(dim * 4, dim)

        self.local_conv = LocalFeatureExtractor(dim, 
                                                kernel_size=3, 
                                                dilation=1)
        
        # Optional: A Gate to let the network choose emphasis
        self.mixer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb=None):
        B, C, H, W = x.shape
        residual = x
        
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)

        if t_emb is not None:
            scale, shift = t_emb.chunk(2, dim=1)
            x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # ---------------------------------------------------------
        # 2. HORIZONTAL SCANS (Raster Order)
        # ---------------------------------------------------------
        # Forward ->
        out_h_fwd = self.mamba_h_fwd(x_norm)
        
        # Backward <-
        x_flip = torch.flip(x_norm, dims=[1])
        out_h_bwd = self.mamba_h_bwd(x_flip)
        out_h_bwd = torch.flip(out_h_bwd, dims=[1]) # Flip back

        # ---------------------------------------------------------
        # 3. VERTICAL SCANS (Column-Major Order)
        # ---------------------------------------------------------
        # Reshape to Image -> Transpose (Swap H and W) -> Flatten
        # Result: (B, W*H, C). Now 'neighbors' in seq are vertical neighbors.
        x_v_img = x_norm.view(B, H, W, C).permute(0, 2, 1, 3) 
        x_v_flat = x_v_img.flatten(1, 2)
        
        # Down v
        out_v_fwd = self.mamba_v_fwd(x_v_flat)
        
        # Up ^
        x_v_flip = torch.flip(x_v_flat, dims=[1])
        out_v_bwd = self.mamba_v_bwd(x_v_flip)
        out_v_bwd = torch.flip(out_v_bwd, dims=[1])
        
        # Un-Transpose Vertical Outputs back to Horizontal Order
        # (B, W*H, C) -> (B, W, H, C) -> (B, H, W, C) -> (B, L, C)
        out_v_fwd = out_v_fwd.view(B, W, H, C).permute(0, 2, 1, 3).flatten(1, 2)
        out_v_bwd = out_v_bwd.view(B, W, H, C).permute(0, 2, 1, 3).flatten(1, 2)
        
        ## ---------------------------------------------------------
        # 4. Global Fusion
        # ---------------------------------------------------------
        # Combine all 4 views of the image
        global_feat = self.fusion_linear(
            torch.cat([out_h_fwd, out_h_bwd, out_v_fwd, out_v_bwd], dim=-1)
        )

        # ---------------------------------------------------------
        # 5. Local Branch (Conv)
        # ---------------------------------------------------------
        # Reshape for Conv2d
        x_img_norm = x_norm.transpose(1, 2).view(B, C, H, W)
        local_feat = self.local_conv(x_img_norm)
        local_feat = local_feat.flatten(2).transpose(1, 2)

        
        # ---------------------------------------------------------
        # 6. Gated Output
        # ---------------------------------------------------------
        combined = torch.cat([global_feat, local_feat], dim=-1)
        z = self.mixer(combined)
        
        fused = global_feat * z + local_feat * (1 - z)
        
        x_out = self.out_proj(fused)
        
        # Reshape to (B, C, H, W) for residual add
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        x_out = self.dropout(x_out)
        
        return residual + x_out
        
        
class GatedFusion(nn.Module):
    """
    Standard Spatial Gating (Version 1).
    Decides 'where' to fuse information pixel-by-pixel.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, 1, 1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, dec_feat, enc_feat):
        # Concatenate and calculate spatial map (B, 1, H, W)
        gate = self.conv(torch.cat([dec_feat, enc_feat], dim=1))
        # Weighted sum based on spatial location
        fused = dec_feat * (1 - gate) + enc_feat * gate
        return self.out_conv(fused)

# --- VERSION 2 COMPONENTS (SOTA) ---
class CAGatedFusion(nn.Module):
    """
    Channel Attention Gating (Version 2).
    Decides 'what features' (texture vs fog) to fuse using Global Context.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Squeeze (Global Context)
            nn.Conv2d(dim * 2, dim // 2, 1),  # Compress
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim * 2, 1),  # Excite
            nn.Sigmoid()                      # Weight
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, dec_feat, enc_feat):
        combined = torch.cat([dec_feat, enc_feat], dim=1)
        weights = self.attn(combined)
        w_dec, w_enc = weights.chunk(2, dim=1)
        # Channel-wise weighted fusion
        fused = (dec_feat * w_dec) + (enc_feat * w_enc)
        return self.conv(fused)



class PixelShuffleUpsample(nn.Module):
    """
    SOTA Trick: Replaces ConvTranspose2d to eliminate checkerboard artifacts.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        # We need to project to (dim_out * 4) so PixelShuffle(2) results in dim_out
        self.conv = nn.Conv2d(dim_in, dim_out * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2) # Scale x2
        
    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))



class FM_PhysMamba_UNET(nn.Module):
    def __init__(self, model_cfg_path="small", 
                       in_channels=3, 
                       use_version=2, 
                       gradient_checkpointing=False):
        super().__init__()
        
        # 1. Load Config
        if isinstance(model_cfg_path, CN):
            self.cfg = model_cfg_path
        else:
            self.cfg = get_model_config(model_cfg_path)

        # 2. Params
        base_dim = self.cfg.BASE_DIM
        dim_mults = self.cfg.DIM_MULTS
        self.physics_guided = self.cfg.PHYSICS_GUIDED
        enc_blocks_list = self.cfg.ENCODER_BLOCKS
        dec_blocks_list = self.cfg.DECODER_BLOCKS

        self.use_version = use_version 
        self.use_checkpoint = gradient_checkpointing # <--- New Flag
        self.dims = [base_dim * m for m in dim_mults]

        num_mid_blocks = enc_blocks_list[-1] if len(enc_blocks_list) >= len(self.dims) else 2

        # --- Time & Physics Embedding ---
        time_dim = base_dim * self.cfg.TIME_DIM_MULT
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.phys_gate = nn.Sequential(
            nn.Linear(time_dim + 3, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.down_time_projs = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()

        # --- ENCODER ---
        self.init_conv = nn.Conv2d(in_channels, self.dims[0], 3, 1, 1)
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(len(self.dims) - 1):
            dim_in, dim_out = self.dims[i], self.dims[i+1]
            self.down_time_projs.append(nn.Linear(time_dim, dim_in * 2))
            
            # Use ModuleList instead of Sequential for checkpointing control
            blocks = nn.ModuleList([PhysConvNeXtBlock(dim_in)])
            num_mamba = enc_blocks_list[i] if i < len(enc_blocks_list) else 1
            for _ in range(num_mamba):
                blocks.append(PhysBiMambaBlock(dim_in))
            
            self.downs.append(blocks)
            self.downsamples.append(nn.Conv2d(dim_in, dim_out, 4, 2, 1)) 


        mid_dim = self.dims[-1]
        self.mid_time_proj = nn.Linear(time_dim, mid_dim * 2)
        self.mid_blocks = nn.ModuleList()
        for _ in range(num_mid_blocks):
            self.mid_blocks.append(PhysBiMambaBlock(mid_dim))
        
        self.atm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid_dim, 64), nn.SiLU(),
            nn.Linear(64, 3), nn.Sigmoid() 
        )

        # --- DECODER ---
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        for idx, i in enumerate(range(len(self.dims)-2, -1, -1)):
            dim_in, dim_out = self.dims[i+1], self.dims[i]
            self.up_time_projs.append(nn.Linear(time_dim, dim_out * 2))
            # self.up_samples.append(nn.ConvTranspose2d(dim_in, dim_out, 2, 2))
            # self.gates.append(GatedFusion(dim_out))

            # --- SWITCHING LOGIC ---
            if self.use_version == 1:
                # Version 1: Standard Deconv + Spatial Gating
                self.up_samples.append(nn.ConvTranspose2d(dim_in, dim_out, 2, 2))
                self.gates.append(GatedFusion(dim_out))
            else:
                # Version 2: PixelShuffle + Channel Attention Gating (SOTA)
                self.up_samples.append(PixelShuffleUpsample(dim_in, dim_out))
                self.gates.append(CAGatedFusion(dim_out))
            # -----------------------
            
            layers = nn.ModuleList()
            num_mamba = dec_blocks_list[i] if i < len(dec_blocks_list) else 1
            for _ in range(num_mamba):
                if i > 0: layers.append(PhysBiMambaBlock(dim_out))
                else: layers.append(PhysConvNeXtBlock(dim_out))
            layers.append(PhysConvNeXtBlock(dim_out)) 
            self.ups.append(layers)

        self.trans_head = nn.Sequential(
            nn.Conv2d(self.dims[0], 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid() 
        )
        self.final_conv = nn.Conv2d(self.dims[0], 3, 1)
        nn.init.constant_(self.trans_head[-2].bias, 1.0)
        
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
        skips = []
        
        # 2. ENCODER
        for i, (block_list, down_layer) in enumerate(zip(self.downs, self.downsamples)):
            t_emb = self.down_time_projs[i](t_vec)
            for layer in block_list:
                if self.use_checkpoint and self.training:
                    h = checkpoint.checkpoint(layer, h, t_emb, use_reentrant=False)
                else:
                    h = layer(h, t_emb)
            skips.append(h)
            h = down_layer(h)
            
        # 3. BOTTLENECK
        t_emb_mid = self.mid_time_proj(t_vec)

        # We will adapt to this later (Maybe for the O-HAZE DENSE-HAZE Training)
        for block in self.mid_blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint.checkpoint(block, h, t_emb_mid, use_reentrant=False)
            else:
                h = block(h, t_emb_mid)
        
        A_pred = self.atm_head(h).view(-1, 3, 1, 1)
        
        if self.use_version == 2:
            phys_cond = torch.cat([t_vec, A_pred.squeeze(-1).squeeze(-1)], dim=-1)
            t_vec = self.phys_gate(phys_cond)

        # 4. DECODER
        for i in range(len(self.ups)):
            h = self.up_samples[i](h) 
            if len(skips) > 0:
                skip = skips.pop()
                h = self.gates[i](h, skip)
            
            block_list = self.ups[i]
            t_emb_dec = self.up_time_projs[i](t_vec)
            
            for layer in block_list:
                if self.use_checkpoint and self.training:
                    h = checkpoint.checkpoint(layer, h, t_emb_dec, use_reentrant=False)
                else:
                    h = layer(h, t_emb_dec)
                    
        t_map = self.trans_head(h)
        if self.physics_guided:
            h = h * (1 + t_map)
            
        v_pred = self.final_conv(h)

        return v_pred, t_map, A_pred


        
