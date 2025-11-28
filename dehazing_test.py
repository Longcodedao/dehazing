# %%
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import v2
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchdiffeq import odeint
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import math
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from data import (
    get_haze_transforms,
    restandardize_tensor,
    print_transform_summary,
    plotting_pair_images,
    partition_dataset,
)
from data import RESIDE_Indoor, Haze4k_Dataset, OHAZE_Dataset, DENSE_Haze_Dataset
from einops import rearrange
import json

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
B1, B2 = 0.5, 0.999
W_FLOW, W_MSE, W_PERC, W_ADV = 1.0, 1.0, 0.1, 0.01
TIME_LIMIT_HOURS = 11.5
BATCH_SIZE = 256


# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# %%
## Training process


resize_size = 256

# Loading the RESIDE Datset
train_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="train", verbose=True
)
val_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="val", verbose=True
)

reside_dataset = RESIDE_Indoor(
    dataset_path="dataset/indoor-training-set", transform=None
)
train_reside_dataset, val_reside_dataset = partition_dataset(
    reside_dataset, train_transform_reside, val_transform_reside, train_ratio=0.8
)


# Loading the Haze4k Dataset
train_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="train", verbose=True
)
val_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="val", verbose=True
)

haze_4k_train = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="train", transform=train_transform_haze4k
)

haze_4k_val = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="val", transform=val_transform_haze4k
)

train_dataset = ConcatDataset([train_reside_dataset, haze_4k_train])
val_dataset = ConcatDataset([val_reside_dataset, haze_4k_val])

# %%
print(f"Length of train dataset: {len(train_dataset)} ")
print(f"Length of valid dataset: {len(val_dataset)}")

# %%
## Loading DenseHaze dataset

transform_densehaze = get_haze_transforms(
    dataset_name="DENSE-HAZE", resize_size=resize_size, split="val", verbose=True
)

dense_haze = DENSE_Haze_Dataset(
    root_dir="dataset/dense-haze", transform=transform_densehaze
)
print("Length of Dense Haze dataset is: ", len(dense_haze))

# %%
transform_ohaze = get_haze_transforms(
    dataset_name="OHAZE", resize_size=resize_size, split="val", verbose=True
)
o_haze = OHAZE_Dataset(root_dir="dataset/o-haze/O-HAZY", transform=transform_densehaze)
print("Length of O Haze dataset is: ", len(o_haze))

# %%
## Loader dataset
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
dense_haze_loader = DataLoader(dense_haze, batch_size=16, shuffle=False, num_workers=4)
o_haze_loader = DataLoader(o_haze, batch_size=16, shuffle=False, num_workers=4)


## Test Loader
first_train_loader = next(iter(train_loader))
clean, hazy = first_train_loader
print(f"Clean batch has shape: {clean.shape}")
print(f"Hazy batch has shape: {hazy.shape}")


# %%
## Learning model to implement
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


batch_size = 32
x = torch.tensor([1, 50, 100, 500])

sinsoid_emb = SinusoidalPosEmb(256)
pos_emb_1 = sinsoid_emb(x)
pos_emb_2 = sinsoid_emb.forward_original(x)

print("Shape of this is: ", sinsoid_emb(x).shape)
print(f"First example 1 is: {pos_emb_1[0][:10]}")
print(f"First example 2 is: {pos_emb_2[0][:10]}")


# %%
def convert_to_embedding(x, n_heads):
    """
    Args
    x is a Tensor has shape (b, n_heads * c, h, w)

    Returns:
    out: Tensor with shape (b, n_heads, h * w, c)
    """
    b, _, h, w = x.shape
    out = rearrange(x, "b (n_heads c) h w -> b n_heads (h w) c", n_heads=n_heads)
    return out


n_heads = 4
dim = 64
q = torch.randn(32, dim * n_heads, 16, 16)
out_q = convert_to_embedding(q, n_heads)

print("Shape of the out q is: ", out_q.shape)


# %%
class ResNetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
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

    def forward(self, x, time_embed=None):
        h = self.block1(x)
        if self.time_mlp is not None and time_embed is not None:
            h = h + self.time_mlp(time_embed)[:, :, None, None]
        h = self.block2(h)
        shortcut_h = self.res_conv(x)
        return h + shortcut_h


# batch_size = 32
# t = torch.rand(batch_size)
# sinusoid_embed = SinusoidalPosEmb(dim=256)
# time_embed = sinusoid_embed.forward_original(t)
#
# x = torch.randn(batch_size, 64, 16, 16)
# resnet_block = ResNetBlock(dim=64, dim_out=128, time_emb_dim=256)
# output = resnet_block(x, time_embed)

# print(f"Shape of the output is: {output.shape}")


# %%
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, groups=8):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
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

        # Input the Attention Mechanism
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Multi-Scale context step
        x1 = self.msc_conv1(x)
        x2 = self.msc_conv2(x)
        x3 = self.msc_conv3(x)
        x4 = self.msc_conv4(x)

        x_merge = torch.concatenate([x1, x2, x3, x4], dim=1)
        x_enhanced = torch.concatenate([x, x_merge], dim=1)
        x_enhanced = self.msc_merge(x_enhanced)

        q, k, v = self.to_qkv(x_enhanced).chunk(3, dim=1)
        q = convert_to_embedding(q, self.heads)
        k = convert_to_embedding(k, self.heads)
        v = convert_to_embedding(v, self.heads)

        q = q * self.scale
        attention = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attention = attention.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)

        return self.to_out(out) + x


x = torch.randn(32, 64, 16, 16)
attn_block = AttentionBlock(dim=64)

result = attn_block(x)
print("Output result of Attention Block is: ", result.shape)


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
    ):
        super().__init__()
        self.res_block1 = ResNetBlock(
            dim_in, dim_out, time_emb_dim=time_embed_dim, groups=groups
        )
        self.res_block2 = ResNetBlock(
            dim_out, dim_out, time_emb_dim=time_embed_dim, groups=groups
        )
        self.attn = (
            AttentionBlock(dim_out, heads=num_heads, dim_head=dim_head, groups=groups)
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


# %%
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
    ):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        self.res_block1 = ResNetBlock(
            dim_out + dim_skip, dim_out, time_emb_dim=time_embed_dim, groups=groups
        )
        self.res_block2 = ResNetBlock(
            dim_out, dim_out, time_emb_dim=time_embed_dim, groups=groups
        )
        self.attn = (
            AttentionBlock(dim_out, heads=num_heads, dim_head=dim_head, groups=groups)
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


# %%
batch_size = 32
t = torch.rand(batch_size)
sinusoid_embed = SinusoidalPosEmb(dim=256)
time_embed = sinusoid_embed.forward_original(t)

print(f"Time embedding has shape: {time_embed.shape}")

x = torch.randn(batch_size, 128, 16, 16)
skip = torch.randn(batch_size, 128, 32, 32)
upblock = UpBlock(dim_in=128, dim_skip=128, dim_out=64)
output = upblock(x, time_embed, skip)
print(f"THe size of output is: {output.shape}")


# %%
class UNet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1, 2, 4, 8)):
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
            self.downs.append(DownBlock(d_in, d_out, attn=use_attn))

        self.mid_block1 = ResNetBlock(list_dims[-1], list_dims[-1], time_emb_dim=256)
        self.mid_attn = AttentionBlock(list_dims[-1])
        self.mid_block2 = ResNetBlock(list_dims[-1], list_dims[-1], time_emb_dim=256)

        reversed_dim = list(reversed(list_dims))
        up_in_out = list(zip(reversed_dim[:-1], reversed_dim[1:]))
        dim_skip = reversed_dim[1:]
        print("Dim skip order is: ", dim_skip)
        for i, (d_in, d_out) in enumerate(up_in_out):
            use_attn = i < 2
            self.ups.append(UpBlock(d_in, dim_skip[i], d_out, attn=use_attn))

        self.final_conv = nn.Sequential(
            ResNetBlock(dim, dim), nn.Conv2d(dim, channels, kernel_size=1)
        )

    def forward(self, x, t):
        """
        Args:
        x: input of the haze image
        t: Timeline

        Returns: out: Clean image
        """
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x, t_emb)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for up in self.ups:
            skip = skips.pop()
            #            print("Skip shape: ", skip.shape)
            #            print("x shape: ", x.shape)
            #            print("Time embed shape: ", t_emb.shape)

            x = up(x, t_emb, skip)
            # print("-------------")

        out = self.final_conv(x)
        return out


device = "cpu"
time_stamp = torch.rand(32).to(device)
# sinusoid = SinusoidalPosEmb(dim=64)
# print(sinusoid(time_stamp).shape)
unet = UNet().to(device)
x = torch.randn(32, 3, 256, 256).to(device)

output = unet(x, time_stamp)
print("Size of the output is: ", output.shape)


# %%
print(unet.ups)


# %%
## Implementing Flow Matching
def path_sampler(x0, x1, t):
    """
    Args:
        t: Timestamp uniformly sampled from [0, 1]: (B,)
        x0: Hazy image
        x1: Target image
    Return:
        x_t: Image transition at time t
        u_t: Velocity constant from x0 to x1
    """
    t = t.reshape(-1, 1, 1, 1)
    x_t = x0 * (1 - t) + x1 * t
    u_t = x1 - x0

    return x_t, u_t


class ODESolver:
    def __init__(self, model, nfe=20):
        self.model = model
        self.nfe = nfe

    def ode_func(self, t, x):
        # 1. Ensure the time 't' is a vector of size (B,)
        # The ODE solver passes 't' as a scaler (if batching is not done internally)
        # We must expand/broadcast it to match the batch size 'x'
        t = t.expand(x.size(0))

        # 2. Call the UNet (self.model)
        # The UNet predicts the velocity field (v_theta) given the time and the
        # image state
        v_theta = self.model(t, x)

        return v_theta

    @torch.no_grad()
    def sample(self, x_init):
        # 1. Define the time span for integration (from 0 to 1, in nfe steps)
        t_span = torch.linspace(0, 1, self.nfe, device=x_init.device)

        # 2. Define the ODE function for the solver to use
        # The solver requires a function (t, x) -> dx/dt
        # We can use the method we just defined:
        ode_func = self.ode_func

        # 3. Perform the ODE integration
        solution = odeint(
            ode_func, x_init, t_span, rtol=1e-5, atol=1e-5, method="euler"
        )
        # 4. The solution is a tensor of shape (NFE, B, C, H, W). We return the last state (t=1)
        return solution[-1]


# %%
# Creating the Loss and the Solver

### Importing the VGG-16 model
# vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET_V1).features
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
print(vgg)
file_json = "vgg16_features.json"
mapping = {}
conv_idx = 1
block = 1

for i, layer in enumerate(vgg):
    if isinstance(layer, torch.nn.Conv2d):
        name = f"conv{block}_{conv_idx}"
        mapping[name] = i
        conv_idx += 1

    elif isinstance(layer, torch.nn.ReLU):
        name = f"relu{block}_{conv_idx - 1}"
        mapping[name] = i
    elif isinstance(layer, torch.nn.MaxPool2d):
        name = f"pool{block}"
        mapping[name] = i
        block += 1
        conv_idx = 1

with open(file_json, "w") as f:
    json.dump(mapping, f)

# %%
a, b, c = torch.randn(32, 64 * 3, 256, 256).chunk(3, dim=1)
print(a.shape)
print(b.shape)
print(c.shape)
# %%
