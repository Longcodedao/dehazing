import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mamba_ssm import Mamba
from einops import rearrange
import math

# Import your config helper
from .config_model import get_model_cfg_defaults
from .unet import get_model_config # Assuming this helper exists from your previous code
from yacs.config import CfgNode as CN

class PhysConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def _forward_impl(self, x, t_emb=None):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
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
        return input + x

    def forward(self, x, t_emb=None):
        if self.use_checkpoint and self.training:
            # Note: t_emb must require grad for this to work perfectly, which it usually does
            return checkpoint.checkpoint(self._forward_impl, x, t_emb, use_reentrant=False)
        return self._forward_impl(x, t_emb)


class PhysBiMambaBlock(nn.Module):
    def __init__(self, dim, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = nn.LayerNorm(dim)
        
        self.mamba_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.fusion_linear = nn.Linear(dim * 2, dim)
        self.fusion_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def _forward_impl(self, x, t_emb=None):
        B, C, H, W = x.shape
        residual = x
        
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)

        if t_emb is not None:
            t_val, _ = t_emb.chunk(2, dim=1)
            x_norm = x_norm + t_val.unsqueeze(1)

        out_fwd = self.mamba_fwd(x_norm)
        
        x_flip = torch.flip(x_norm, dims=[1])
        out_bwd = self.mamba_bwd(x_flip)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        z = self.fusion_gate(combined)
        x_fused = self.fusion_linear(combined)
        x_out = x_fused * z
        
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        return residual + x_out

    def forward(self, x, t_emb=None):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, t_emb, use_reentrant=False)
        return self._forward_impl(x, t_emb)


class FM_PhysMamba_UNET(nn.Module):
    def __init__(self, model_cfg_path="small", in_channels=3, gradient_checkpointing=False):
        super().__init__()
        
        # ... (Config loading code same as before) ...
        # For brevity, assuming self.cfg is loaded here:
        if isinstance(model_cfg_path, str):
             self.cfg = get_model_config(model_cfg_path)
        else:
             self.cfg = model_cfg_path

        base_dim = self.cfg.BASE_DIM
        dim_mults = self.cfg.DIM_MULTS
        self.physics_guided = self.cfg.PHYSICS_GUIDED
        enc_blocks_list = self.cfg.ENCODER_BLOCKS
        dec_blocks_list = self.cfg.DECODER_BLOCKS
        
        self.dims = [base_dim * m for m in dim_mults]
        
        # Save the flag
        self.ckpt = gradient_checkpointing 
        
        # --- Time Embedding ---
        time_dim = base_dim * self.cfg.TIME_DIM_MULT
        self.time_mlp = nn.Sequential(
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_time_projs = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()
        self.init_conv = nn.Conv2d(in_channels, self.dims[0], 3, 1, 1)
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # --- ENCODER ---
        for i in range(len(self.dims)-1):
            dim_in, dim_out = self.dims[i], self.dims[i+1]
            self.down_time_projs.append(nn.Linear(time_dim, dim_in * 2))

            blocks = []
            # Pass checkpoinitng flag to blocks
            blocks.append(PhysConvNeXtBlock(dim_in, use_checkpoint=self.ckpt)) 
            
            num_mamba = enc_blocks_list[i] if i < len(enc_blocks_list) else 1
            for _ in range(num_mamba):
                blocks.append(PhysBiMambaBlock(dim_in, use_checkpoint=self.ckpt))
            
            self.downs.append(nn.Sequential(*blocks))
            self.downsamples.append(nn.Conv2d(dim_in, dim_out, 4, 2, 1))

        # --- BOTTLENECK ---
        mid_dim = self.dims[-1]
        self.mid_time_proj = nn.Linear(time_dim, mid_dim * 2)
        self.mid_block1 = PhysBiMambaBlock(mid_dim, use_checkpoint=self.ckpt)
        self.mid_block2 = PhysBiMambaBlock(mid_dim, use_checkpoint=self.ckpt)
        
        self.atm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid_dim, 64), nn.SiLU(), nn.Linear(64, 3), nn.Sigmoid() 
        )

        # --- DECODER ---
        self.ups = nn.ModuleList()
        for idx, i in enumerate(range(len(self.dims)-2, -1, -1)):
            dim_in, dim_out = self.dims[i+1], self.dims[i]
            self.up_time_projs.append(nn.Linear(time_dim, dim_out * 2))
            
            layers = []
            layers.append(nn.ConvTranspose2d(dim_in, dim_out, 2, 2))
            layers.append(nn.Conv2d(dim_out*2, dim_out, 1))
            
            num_mamba = dec_blocks_list[i] if i < len(dec_blocks_list) else 1
            for _ in range(num_mamba):
                if i > 0:
                    layers.append(PhysBiMambaBlock(dim_out, use_checkpoint=self.ckpt))
                else:
                    layers.append(PhysConvNeXtBlock(dim_out, use_checkpoint=self.ckpt))
            
            layers.append(PhysConvNeXtBlock(dim_out, use_checkpoint=self.ckpt)) 
            self.ups.append(nn.Sequential(*layers))

        self.trans_head = nn.Sequential(
            nn.Conv2d(self.dims[0], 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid() 
        )
        self.final_conv = nn.Conv2d(self.dims[0], 3, 1)

    def get_sinusoidal_emb(self, t, device):
        half_dim = self.dims[0] // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t):
        t_emb_raw = self.get_sinusoidal_emb(t, x.device) 
        t_vec = self.time_mlp(t_emb_raw)
        
        h = self.init_conv(x)
        skips = []
        
        for i, (block_stack, down_layer) in enumerate(zip(self.downs, self.downsamples)):
            t_emb = self.down_time_projs[i](t_vec)
            for layer in block_stack:
                if isinstance(layer, (PhysConvNeXtBlock, PhysBiMambaBlock)):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            h = down_layer(h)
            
        t_emb_mid = self.mid_time_proj(t_vec)
        h = self.mid_block1(h, t_emb_mid)
        h = self.mid_block2(h, t_emb_mid)
        A_pred = self.atm_head(h).view(-1, 3, 1, 1)
        
        for i, block_stack in enumerate(self.ups):
            h = block_stack[0](h) 
            if len(skips) > 0:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            h = block_stack[1](h) 
            t_emb = self.up_time_projs[i](t_vec)
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