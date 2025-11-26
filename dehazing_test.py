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
