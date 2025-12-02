import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class SinusoidalPosEmb(nn.Module):
    """
    Note that the implementation is a little bit different from the
    Transformers paper but when fed in the linear layers, they learn the weighted
    sums accross the entire input vector
    => All positional information are fully encoded
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        exponential_denominator = half_dim - 1
        log_base = math.log(10000.0)
        c = log_base / (exponential_denominator)

        # Frequencies = 1 / (10000 ^ (i / (d_model // 2 - 1))
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -c)

        # Apply the frequencies to the input scaler (t)
        # t has shape (batch_size, 1) and frequencies has shape (1, half_dim)
        arguments = x[:, None] * frequencies[None, :]

        return torch.cat((arguments.sin(), arguments.cos()), dim=-1)

    def forward_original(self, x):
        device = x.device
        half_dim = self.dim // 2

        exponential_denominator = half_dim - 1
        log_base = math.log(10000.0)
        c = log_base / (exponential_denominator)

        # Frequencies = 1 / (10000 ^ (i / (d_model // 2 - 1))
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -c)

        arguments = x[:, None] * frequencies[None, :]

        sin_component = arguments.sin()
        cos_component = arguments.cos()

        stacked = torch.stack((sin_component, cos_component), dim=-1)

        # Reshape(batch_size, half_dim, 2) to (batch_size, half_dim * 2)
        # Collapsing the final dimension and let those elements interleaved together
        interleaved_eb = stacked.view(x.shape[0], -1)
        return interleaved_eb


class ResNetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.time_mlp = (
            nn.Sequential(nn.Linear(time_emb_dim, dim_out), nn.SiLU())
            if time_emb_dim
            else None
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward_impl(self, x, time_embed=None):
        h = self.block1(x)
        if self.time_mlp is not None and time_embed is not None:
            h = h + self.time_mlp(time_embed)[:, :, None, None]
        h = self.block2(h)
        shortcut_h = self.res_conv(x)

        return h + shortcut_h

    def forward(self, x, time_embed=None):
        if self.training and self.use_checkpoint:
            # Checkpointing requires inputs to require_grad for backward to work correctly
            # Often x requires grad, but time_embed might not.
            return checkpoint.checkpoint(
                self.forward_impl, x, time_embed, use_reentrant=False
            )
        else:
            return self.forward_impl(x, time_embed)


class AttentionBlock(nn.Module):
    def __init__(
        self, dim, heads=4, dim_head=32, groups=8, patch_size=8, use_checkpoint=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        return x, pad_r, pad_b

    def forward_impl(self, x):
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

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self.forward_impl, x, use_reentrant=False)
        else:
            return self.forward_impl(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        attn=False,
        time_embed_dim=256,
        num_heads=4,
        dim_head=32,
        groups=8,
        use_checkpoint=False,
    ):
        super().__init__()
        self.res_block1 = ResNetBlock(
            dim_in,
            dim_out,
            time_emb_dim=time_embed_dim,
            groups=groups,
            use_checkpoint=use_checkpoint,
        )
        self.res_block2 = ResNetBlock(
            dim_out,
            dim_out,
            time_emb_dim=time_embed_dim,
            groups=groups,
            use_checkpoint=use_checkpoint,
        )
        self.attn = (
            AttentionBlock(
                dim_out,
                heads=num_heads,
                dim_head=dim_head,
                groups=groups,
                use_checkpoint=use_checkpoint,
            )
            if attn
            else nn.Identity()
        )

        self.downsample = nn.Conv2d(
            dim_out, dim_out, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, t_emb):
        x = self.res_block1(x, t_emb)
        x = self.attn(x)
        x = self.res_block2(x, t_emb)
        x = self.downsample(x)

        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_skip,
        dim_out,
        attn=False,
        time_embed_dim=256,
        num_heads=4,
        dim_head=32,
        groups=8,
        use_checkpoint=False,
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        self.res_block1 = ResNetBlock(
            dim_out + dim_skip,
            dim_out,
            time_emb_dim=time_embed_dim,
            groups=groups,
            use_checkpoint=use_checkpoint,
        )
        self.res_block2 = ResNetBlock(
            dim_out,
            dim_out,
            time_emb_dim=time_embed_dim,
            groups=groups,
            use_checkpoint=use_checkpoint,
        )
        self.attn = (
            AttentionBlock(
                dim_out,
                heads=num_heads,
                dim_head=dim_head,
                groups=groups,
                use_checkpoint=use_checkpoint,
            )
            if attn
            else nn.Identity()
        )

    def forward(self, x, time_embed, skip):
        x = self.upsample(x)
        x = self.conv(x)
        # Add the SKip connection from the Down layers
        x = torch.concatenate([x, skip], dim=1)
        # print("After Concatenating: ", x.shape)
        x = self.res_block1(x, time_embed)
        x = self.attn(x)
        x = self.res_block2(x, time_embed)

        return x


class UNet(nn.Module):
    def __init__(
        self, dim=64, channels=3, dim_mults=(1, 2, 4, 8), use_checkpoint=False
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(channels, dim, kernel_size=7, padding=3)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, 256),
        )

        list_dims = [dim * m for m in dim_mults]
        list_dims = [dim] + list_dims
        in_out = list(zip(list_dims[:-1], list_dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for i, (d_in, d_out) in enumerate(in_out):
            use_attn = i >= 2
            self.downs.append(
                DownBlock(d_in, d_out, attn=use_attn, use_checkpoint=use_checkpoint)
            )

        self.mid_block1 = ResNetBlock(
            list_dims[-1],
            list_dims[-1],
            time_emb_dim=256,
            use_checkpoint=use_checkpoint,
        )
        self.mid_attn = AttentionBlock(list_dims[-1], use_checkpoint=use_checkpoint)
        self.mid_block2 = ResNetBlock(
            list_dims[-1],
            list_dims[-1],
            time_emb_dim=256,
            use_checkpoint=use_checkpoint,
        )

        reversed_dim = list(reversed(list_dims))
        up_in_out = list(zip(reversed_dim[:-1], reversed_dim[1:]))
        dim_skip = reversed_dim[1:]

        for i, (d_in, d_out) in enumerate(up_in_out):
            use_attn = i < 2
            self.ups.append(
                UpBlock(
                    d_in,
                    dim_skip[i],
                    d_out,
                    attn=use_attn,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.final_conv = nn.Sequential(
            ResNetBlock(dim, dim, use_checkpoint=use_checkpoint),
            nn.Conv2d(dim, channels, kernel_size=1),
        )

    def forward(self, x, t, profiler=None):
        """
        Args:
        x: input of the haze image
        t: Timeline

        Returns: out: Clean image
        """
        if profiler:
            profiler.print_status("  [UNet] Start")

        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        if profiler:
            profiler.print_status("  [UNet] Init Conv")

        skips = []
        for i, down in enumerate(self.downs):
            skips.append(x)
            x = down(x, t_emb)
            if profiler:
                profiler.print_status(f"  [UNet] Down {i}")

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        if profiler:
            profiler.print_status("  [UNet] Mid Block")

        for up in self.ups:
            skip = skips.pop()
            x = up(x, t_emb, skip)
            if profiler:
                profiler.print_status(f"  [UNet] Up {i}")
            # print("-------------")

        out = self.final_conv(x)
        if profiler:
            profiler.print_status("  [UNet] End")
        return out
