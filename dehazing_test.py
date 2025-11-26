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
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from data import (
    get_haze_transforms,
    restandardize_tensor,
    print_transform_summary,
    plotting_pair_images,
)
from data import RESIDE_Indoor, Haze4k_Dataset, OHAZE_Dataset, DENSE_Dataset

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
B1, B2 = 0.5, 0.999
W_FLOW, W_MSE, W_PERC, W_ADV = 1.0, 1.0, 0.1, 0.01
TIME_LIMIT_HOURS = 11.5


# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# %%
## visualization
# def plotting_pair_images(dataset, num_instances=3, start_index=0, save_figure=False):
#    N_COLS = 2
#    N_ROWS = num_instances
#
#    end_index = start_index + num_instances
#    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
#    fig.suptitle(
#        f"GT vs Haze Image Comparision in {dataset}",
#        fontsize=16,
#    )
#    row_index = 0
#    for i in range(start_index, end_index):
#        clean, hazy = dataset[i]
#        clean = restandardize_tensor(clean)
#        hazy = restandardize_tensor(hazy)
#        clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
#        axes[row_index][0].imshow(clean_display)
#        axes[row_index][0].set_title(f"Clean {i}")
#        axes[row_index][0].axis("off")
#
#        axes[row_index][1].imshow(hazy_display)
#        axes[row_index][1].set_title(f"Hazy {i}")
#        axes[row_index][1].axis("off")
#
#        row_index += 1
#
#    save_path = f"images/{dataset}_hazy_clear_comparison.png"
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
#
#    if save_figure:
#        print(f"Saving visualization to: {save_path}")
#        plt.savefig(
#            save_path, dpi=300, bbox_inches="tight"
#        )  # Saves the figure with high resolution and tight bounds
#
#    plt.show()


# %%
resize_size = 256

print("Training Transform is:")
train_transform = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="train", verbose=True
)
reside_dataset = RESIDE_Indoor(
    dataset_path="dataset/indoor-training-set", transform=train_transform
)
plotting_pair_images(reside_dataset, save_figure=True)


# %%
## Testing loading Haze4K Dataset with transforms
train_transform = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=256, split="train", verbose=True
)

haze4k_dataset = Haze4k_Dataset(
    root_dir=Path("dataset/haze4k"), split="train", transform=train_transform
)
plotting_pair_images(haze4k_dataset, save_figure=True)


# %%
## Loading the O-HAZE Dataset with transforms
val_transform = get_haze_transforms(
    dataset_name="O-HAZE", resize_size=256, split="val", verbose=True
)

o_haze_path = Path("dataset/o-haze/O-HAZY/")
o_haze_dataset = OHAZE_Dataset(root_dir=o_haze_path, transform=val_transform)
plotting_pair_images(o_haze_dataset, save_figure=True)

# %%
## Loading the Dense Haze dataset
densehaze_path = Path("dataset/dense-haze/")

train_transform = get_haze_transforms(
    dataset_name="DENSE-HAZE", resize_size=256, split="val", verbose=True
)
dense_haze = DENSE_Dataset(root_dir=densehaze_path, transform=train_transform)

plotting_pair_images(dense_haze, save_figure=True)
# %%
