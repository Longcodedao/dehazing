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
from data import RESIDE_Indoor, get_haze_transforms


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
### Starting EDA the dataset
indoor_data = Path("dataset/indoor-training-set")
sub_dir = sorted(list(indoor_data.glob("*")))
print(sub_dir)

clean_dir = indoor_data / "clear"
hazy_dir = indoor_data / "hazy"

hazy_images_dir = sorted(list(hazy_dir.glob("*")))
print(hazy_images_dir[0])
print("Total images in the Hazy path is: ", len(hazy_images_dir))

clean_images_dir = sorted(list(clean_dir.glob("*")))
print(clean_images_dir[0])
print("Total images in the Hazy path is: ", len(clean_images_dir))

# %%
metadata_path = indoor_data / "metadata.csv"
metadata_csv = pd.read_csv(metadata_path)

# %%
first_instance = metadata_csv.iloc[0]
clear_image = first_instance["clear_image_path"]
hazy_images = first_instance["hazy_image_paths"].strip("[]").replace("'", "").split(",")

print("Clear Image: ", clear_image)
print("Hazy Images: ", hazy_images)
clear_image = indoor_data / clear_image
clear_image = np.array(Image.open(clear_image).convert("RGB"))

hazy_images_path = [indoor_data / hazy_image.strip() for hazy_image in hazy_images]
print(hazy_images_path)
hazy_images = [Image.open(hazy_image).convert("RGB") for hazy_image in hazy_images_path]
hazy_images = [np.array(hazy_image) for hazy_image in hazy_images]

num_hazy = len(hazy_images)
num_total = num_hazy + 1

N_ROWS = 2
N_COLS = math.ceil(num_total / N_ROWS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle("Clear (Ground Truth) and Hazy Image Comparison", fontsize=16)

axes = axes.flatten()

# 3. Prepare all images and titles for display
all_images = [clear_image] + hazy_images
all_titles = ["Clear Image (Ground Truth)"] + [
    f"Hazy Image {i + 1}" for i in range(num_hazy)
]

# 4. Display all images using a single loop
for i in range(num_total):
    axes[i].imshow(all_images[i])
    axes[i].set_title(all_titles[i], fontsize=10)
    axes[i].axis("off")  # Hide axis ticks and labels

# 5. Hide any unused subplots
# If N_ROWS * N_COLS is greater than num_total (which happens when num_total is odd),
# we need to hide the blank plots at the end.
for j in range(num_total, N_ROWS * N_COLS):
    fig.delaxes(axes[j])

# 6. Show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
plt.show()


# %%
resize_size = 256

print("Training Transform is:")
train_transform = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="train", verbose=True
)
reside_dataset = RESIDE_Indoor(
    dataset_path="dataset/indoor-training-set", transform=train_transform
)
first_instance = reside_dataset[0]
clean, hazy = first_instance

print(f"Clean image has shape: {clean.shape}")
print(f"Hazy image has shape: {hazy.shape}")

# %%
N_COLS = 2
images_display = 5
N_ROWS = images_display


fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle("Clear (Ground Truth) and Hazy Image Comparison", fontsize=16)

for i in range(images_display):
    clean, hazy = reside_dataset[i]
    clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
    axes[i][0].imshow(clean_display)
    axes[i][0].set_title(f"Clean {i}")
    axes[i][0].axis("off")

    axes[i][1].imshow(hazy_display)
    axes[i][1].set_title(f"Hazy {i}")
    axes[i][1].axis("off")

# 6. Show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
plt.show()

# %%
