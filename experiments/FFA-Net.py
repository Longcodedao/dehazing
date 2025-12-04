# %%
import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Dict
from torch.utils.flop_counter import FlopCounterMode
from data.utils import print_transform_summary
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import os
import pandas as pd
import numpy as np


# %%
# Getting the CHannel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=1):
        super().__init__()

        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(
                dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PixelAttention(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2

        self.pa = nn.Sequential(
            nn.Conv2d(
                dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, 1, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# %%
## FFA Block Module
class FFA_Block(nn.Module):
    def __init__(self, dim, kernel_size, bias=True):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )

        # Pixel Attention
        hidden_dim = dim // 8
        self.pixel_attention = PixelAttention(dim, hidden_dim, kernel_size=1)
        self.channel_attention = ChannelAttention(dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res += x

        res = self.conv2(res)
        res = self.channel_attention(res)
        res = self.pixel_attention(res)

        res = res + x

        return res


# %%
# Group Block
class Group(nn.Module):
    def __init__(self, dim, kernel_size, blocks, bias=True):
        super().__init__()
        self.blocks = blocks
        modules = [FFA_Block(dim, kernel_size) for _ in range(blocks)]
        modules.append(
            nn.Conv2d(
                dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
            )
        )
        self.group = nn.Sequential(*modules)

    def forward(self, x):
        res = self.group(x)
        res += x
        return res


# %%
# Create the FFA-Net Model
class FFA(nn.Module):
    def __init__(
        self, groups, blocks, channels=3, dim=64, kernel_size=3, padding=1, bias=True
    ):
        super().__init__()
        self.num_groups = groups

        self.preprocess = nn.Conv2d(
            channels, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.groups = nn.ModuleList(
            [Group(dim, kernel_size=kernel_size, blocks=blocks) for _ in range(groups)]
        )
        self.channel_attention = ChannelAttention(
            dim * groups,
            dim // 16,
        )

        self.pixel_attention = PixelAttention(dim, dim // 8)

        self.postprocess = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.Conv2d(dim, 3, kernel_size=kernel_size, padding=padding, bias=bias),
        )

    def forward(self, x):
        x1 = self.preprocess(x)
        res = []

        temp = x1
        for idx, layer in enumerate(self.groups):
            out_group = layer(temp)
            res.append(out_group)
            temp = out_group

        #        # Shape [Batch, Groups, Channel, H, W]
        #        res = torch.stack(res, dim=1)
        #
        #        # Applying Channel Attention for all of the groups
        #        w_in = rearrange(res, "b g c h w -> b (g c) h w")
        #        weighted_res = self.channel_attention(w_in)
        #        # weighted_res' already contains (w1*res1, w2*res2, ...) concatenated.
        #        weighted_res = rearrange(
        #            weighted_res, "b (g c) h w -> b g c h w", g=self.num_groups
        #        )

        # Shape [Batch, Groups * CHannels, H, W]
        res = torch.cat(res, dim=1)
        weighted_res = self.channel_attention(res)
        weighted_res = rearrange(
            weighted_res, "b (g c) h w -> b g c h w", g=self.num_groups
        )

        # This is mathematically equivalent to: w1 * res1 + w2 * res2 + w3* res3
        out = torch.sum(weighted_res, dim=1)

        # Apply Pixel Attention
        out = self.pixel_attention(out)

        x_out = self.postprocess(out)

        return x + x_out


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SHAPE = (1, 3, 240, 240)
x = torch.randn(INPUT_SHAPE).to(device)
ffa_net = FFA(groups=3, blocks=19).to(device)

out = ffa_net(x)
print(f"Shape of the output is: {out.shape}")

# %%
print(f"Calculating FLOPs for input shape: {INPUT_SHAPE}")

with FlopCounterMode(ffa_net) as flop_counter:
    _ = ffa_net(x)

# Get the total FLOPs and format the output
total_flops = flop_counter.get_total_flops()

# Convert FLOPs to Giga-FLOPs (GFLOPs) for a more readable number
gflops = total_flops / 10**9


# %%
## Get the total parameters
total_params = sum(p.numel() for p in ffa_net.parameters() if p.requires_grad)
n_params = total_params / (10**6)

print("-" * 40)
print(f"Total FLOPs: {total_flops:,.0f}")
print(f"Total GFLOPs: {gflops:.3f} G")
print(f"Total parameters (in M): {n_params:.3f}M")
print("-" * 40)

# %%
## Geting the datasets
from pathlib import Path

resolution = 240
batch_size = 32
dataset_root = "/kaggle/input/"
reside_indoor_path = "indoor-training-set-its-residestandard"
reside_sots_path = "synthetic-objective-testing-set-sots-reside"
verbose = True


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
                data_item = {"index": idx, "clean": clean_path, "hazy": hazy_path}

                self.data.append(data_item)

    def __repr__(self):
        return "RESIDE Indoor"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Missing image file at {clean_path} or {hazy_path}. Skipping")
            return self.__getitem__((idx + 1) % len(self))

        # Check if the size differs so that we can crop to have the same size
        w_clean, h_clean = clean_img.size
        w_hazy, h_hazy = hazy_img.size

        if clean_img.size != hazy_img.size:
            common_w = min(w_clean, w_hazy)
            common_h = min(h_clean, h_hazy)

            clean_img = v2.CenterCrop(size=(common_w, common_h))(clean_img)
            hazy_img = v2.CenterCrop(size=(common_w, common_h))(hazy_img)

        if self.transform:
            clean_img, hazy_img = self.transform(clean_img, hazy_img)
        else:
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )

        return clean_img, hazy_img


class RESIDE_SOTS_Indoor(Dataset):
    def __init__(self, dataset_path, transform=None, metadata="metadata.csv"):
        self.root_dir = Path(dataset_path)
        self.metadata_csv = pd.read_csv(self.root_dir / metadata)

        self.transform = transform
        self.data = []
        for idx, row in self.metadata_csv.iterrows():
            clean_path = self.root_dir / "indoor" / row["clear_image_path"]
            hazy_paths_str = row["hazy_image_paths"]
            hazy_image_paths = [
                path.strip()
                for path in hazy_paths_str.strip("[]").replace("'", "").split(",")
            ]
            list_hazy_paths = [
                self.root_dir / "indoor" / hazy_path for hazy_path in hazy_image_paths
            ]
            for hazy_path in list_hazy_paths:
                data_item = {"index": idx, "clean": clean_path, "hazy": hazy_path}

                self.data.append(data_item)

    def __repr__(self):
        return "RESIDE Indoor"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Missing image file at {clean_path} or {hazy_path}. Skipping")
            return self.__getitem__((idx + 1) % len(self))

        # Check if the size differs so that we can crop to have the same size
        w_clean, h_clean = clean_img.size
        w_hazy, h_hazy = hazy_img.size

        if clean_img.size != hazy_img.size:
            common_w = min(w_clean, w_hazy)
            common_h = min(h_clean, h_hazy)

            clean_img = v2.CenterCrop(size=(common_w, common_h))(clean_img)
            hazy_img = v2.CenterCrop(size=(common_w, common_h))(hazy_img)

        if self.transform:
            clean_img, hazy_img = self.transform(clean_img, hazy_img)
        else:
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )

        return clean_img, hazy_img


def get_haze_transforms(
    resize_size,
    dataset_name="RESIDE",
    split="train",
    verbose=True,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
):
    # --- Component Definitions ---
    common_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    if split == "train":
        # Discrete Rotation: 0, 90, 180, 270 (Legacy behavior)
        discrete_rotation = v2.RandomChoice(
            [
                v2.Identity(),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ]
        )

        geometric_sync_transformations = v2.Compose(
            [
                v2.RandomCrop(resize_size, pad_if_needed=True, padding_mode="reflect"),
                v2.RandomHorizontalFlip(p=0.5),
                discrete_rotation,
            ]
        )

        def train_transform(clean_img, hazy_img):
            clean_img, hazy_img = geometric_sync_transformations(clean_img, hazy_img)

            clean_img = common_transforms(clean_img)
            hazy_img = common_transforms(hazy_img)

            return clean_img

        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tTrain Mode (Size: {resize_size}x{resize_size})",
                geometric_sync=geometric_sync_transformations,
                haze_only=v2.Identity(),
                common=common_transforms,
            )

        return train_transform

    else:

        def val_transform(clean_img, hazy_img):
            # 2. Common Transforms (Format & Normalize)
            clean_img = common_transforms(clean_img)
            hazy_img = common_transforms(hazy_img)

            return hazy_img, clean_img

        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tEval Mode (Size: {resize_size}x{resize_size})",
                geometric_sync=v2.Identity(),
                haze_only=v2.Identity(),
                common=common_transforms,
            )

        return val_transform


# %%

train_transform_reside = get_haze_transforms(
    dataset_name="RESIDE",
    resize_size=resolution,
    split="train",
    verbose=verbose,
    mean=[0.64, 0.6, 0.58],
    std=[0.14, 0.15, 0.152],
)

val_transform_reside = get_haze_transforms(
    dataset_name="RESIDE",
    resize_size=resolution,
    split="val",
    verbose=verbose,
    mean=[0.64, 0.6, 0.58],
    std=[0.14, 0.15, 0.152],
)
train_dataset = RESIDE_Indoor(
    dataset_path=os.path.join(dataset_root, reside_indoor_path),
    transform=train_transform_reside,
)
val_dataset = RESIDE_SOTS_Indoor(
    dataset_path=os.path.join(dataset_root, reside_sots_path),
    transform=val_transform_reside,
    metadata="metadata_indoor.csv",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# %%
import torchvision
from PIL import Image

clear_path = "/kaggle/input/indoor-training-set-its-residestandard/clear/1.png"
hazy_path = "/kaggle/input/indoor-training-set-its-residestandard/hazy/1_1_0.90179.png"

hazy = Image.open(hazy_path)
clear = Image.open(clear_path)

print("Shape of hazy is: ", hazy.size)

clear = torchvision.transforms.CenterCrop(hazy.size[::-1])(clear)
print(clear.size)
print(hazy.size)

# %%
