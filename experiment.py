# %%
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


# %%


class PatchAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, groups=8, patch_size=8):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        self.patch_size = patch_size
        hidden_dim = heads * dim_head

        # Add Multi-Scale Context
        # We are going to diversify the context field by embedding
        # multi convolution with different dilations
        internal_dim = dim // 4
        self.msc_conv1 = nn.Conv2d(
            dim, internal_dim, kernel_size=3, padding=1, dilation=1
        )
        self.msc_conv2 = nn.Conv2d(
            dim, internal_dim, kernel_size=3, padding=2, dilation=2
        )
        self.msc_conv3 = nn.Conv2d(
            dim, internal_dim, kernel_size=3, padding=4, dilation=4
        )
        self.msc_conv4 = nn.Conv2d(
            dim, dim - (3 * internal_dim), kernel_size=3, padding=1, dilation=1
        )
        self.msc_merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1), nn.GroupNorm(groups, dim), nn.SiLU()
        )

        # --- Patch Embedding Projections
        # Fllatened Patch Dimension = C * P * P
        patch_dim = dim * patch_size * patch_size

        # Project patch -> Embedding (dim)
        self.patch_to_emb = nn.Linear(patch_dim, dim)

        # Project Embedding -> Patch
        self.emb_to_patch = nn.Linear(dim, patch_dim)

        # --- Global Attention Projection ---
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def paddding(self, x):
        """Ensure (H, W) of the image x are divisible by patch_size"""
        h, w = x.shape[-2:]

        pad_l = pad_t = 0
        pad_r = (self.patch_size - w % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - h % self.patch_size) % self.patch_size

        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        return x, pad_r, pad_b

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Multi-Scale
        x1 = self.msc_conv1(x)
        x2 = self.msc_conv2(x)
        x3 = self.msc_conv3(x)
        x4 = self.msc_conv4(x)

        x_merge = torch.concatenate([x1, x2, x3, x4], dim=1)
        x_enhanced = torch.concatenate([x, x_merge], dim=1)
        x_enhanced = self.msc_merge(x_enhanced)

        # Apply padding to ensure the H, W is the multiple of self.patch_size
        x_enhanced, pad_r, pad_b = self.paddding(x_enhanced)

        Hp, Wp = x_enhanced.shape[-2:]
        # 2. Patch partition & Platten
        # (B, C, H, W) -> (B, Num_Patches, Patch_Dim)
        # Patch_Dim = C * P * P
        x_patches = rearrange(
            x_enhanced,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        # 3. Patch Embedding (Projection to 'dim')
        # (B, N, C * P * P) -> (B, N, C)
        x_emb = self.patch_to_emb(x_patches)
        # 4. Global Attention on Patches
        qkv = self.to_qkv(x_emb).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = q * self.scale
        attention = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        attention = attention.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)

        # Merge head
        out = rearrange(out, "b h n d -> b n (h d)")

        # 5. Output Projection & Un-Embedding
        out = self.to_out(out)
        out = self.emb_to_patch(out)

        # 6. Un-Patchify (Reshape back to image)
        # (B, H/P * W/P, C*P*P) -> (B, C, H, W)
        out = rearrange(
            out,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=Hp // self.patch_size,
            w=Wp // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # 7. Remove padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w]

        return out + x


# %%
device = torch.device("cuda:3")
x = torch.randn(64, 128, 128, 128).to(device)
patch_attn = PatchAttentionBlock(dim=128).to(device)

out = patch_attn(x)
print(f"Output Shape is: {out.shape}")

# %%
