# %%
import torch
import torch.nn as nn
from model.unet import *


# %%
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

        skip_feature = x
        x = self.downsample(x)

        return x, skip_feature


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
        print("List dim: ", list_dims)
        reversed_dim = list(reversed(list_dims))
        up_in_out = list(zip(reversed_dim[:-1], reversed_dim[1:]))

        print("Reverse dim: ", reversed_dim)
        for i, (d_in, d_out) in enumerate(up_in_out):
            use_attn = i < 2
            self.ups.append(
                UpBlock(
                    d_in,
                    d_in,
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
            x, skip_feat = down(x, t_emb)
            print(f"Index {i}:")
            print(f"Shape of x is: {x.shape}")
            print(f"Shape of skip_feature is: {skip_feat.shape}")
            skips.append(skip_feat)
            if profiler:
                profiler.print_status(f"  [UNet] Down {i}")

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        if profiler:
            profiler.print_status("  [UNet] Mid Block")

        skips = skips[::-1]
        print("\nStart UpConvolution")
        for up in self.ups:
            skip = skips.pop(0)
            print(f"Shape of x is: {x.shape}")
            print(f"Shape of skip_feature is: {skip.shape}")
            x = up(x, t_emb, skip)
            if profiler:
                profiler.print_status(f"  [UNet] Up {i}")
            # print("-------------")

        out = self.final_conv(x)
        if profiler:
            profiler.print_status("  [UNet] End")
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
t = torch.rand(1)
x = torch.randn(1, 3, 256, 256)
model = UNet()
output = model(x, t)

print(output)

# %%
