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
from data import get_haze_transforms, restandardize_tensor, print_transform_summary
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
images_display = 4
N_ROWS = images_display
start_index = 10
end_index = start_index + images_display

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle(
    "Clear (Ground Truth) and Hazy Image Comparison (In the RESIDE Datasets)",
    fontsize=16,
)

row_index = 0
for i in range(start_index, end_index):
    clean, hazy = reside_dataset[i]
    clean = restandardize_tensor(clean)
    hazy = restandardize_tensor(hazy)
    clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
    axes[row_index][0].imshow(clean_display)
    axes[row_index][0].set_title(f"Clean {i}")
    axes[row_index][0].axis("off")

    axes[row_index][1].imshow(hazy_display)
    axes[row_index][1].set_title(f"Hazy {i}")
    axes[row_index][1].axis("off")

    row_index += 1
# 6. Save the plot
save_path = "images/reside_hazy_clear_comparison.png"
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

# 7. Add the save command
print(f"Saving visualization to: {save_path}")
plt.savefig(
    save_path, dpi=300, bbox_inches="tight"
)  # Saves the figure with high resolution and tight bounds

# 8. Show the plot
plt.show()


# %%
### EDA the Haze4K dataset
haze4k_path = Path("dataset/haze4k/Haze4K-T")
clear_path = haze4k_path / "GT"
haze_path = haze4k_path / "IN"

clear_img_path = sorted(list(clear_path.glob("*.png")))
haze_img_path = sorted(list(haze_path.glob("*.png")))
index_image = 10

clear_img = Image.open(clear_img_path[index_image]).convert("RGB")
haze_img = Image.open(haze_img_path[index_image]).convert("RGB")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle(
    "Haze4k Image Comparison",
    fontsize=16,
)
axes = axes.flatten()

axes[0].imshow(clear_img)
axes[0].axis("off")
axes[0].set_title("Clear Image")

axes[1].imshow(haze_img)
axes[1].axis("off")
axes[1].set_title("Haze Image")

save_path = "images/haze4k_comparison.png"
plt.savefig(
    save_path, dpi=300, bbox_inches="tight"
)  # Saves the figure with high resolution and tight bounds
plt.show()


# %%
## Testing loading Haze4K Dataset with transforms
train_transform = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=256, split="train", verbose=True
)

haze_dataset = Haze4k_Dataset(
    root_dir=Path("dataset/haze4k"), split="train", transform=train_transform
)

N_COLS = 2
images_display = 4
N_ROWS = images_display
start_index = 10
end_index = start_index + images_display

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle(
    "Clear (Ground Truth) and Hazy Image Comparison (In the Haze4K Dataset)",
    fontsize=16,
)

row_index = 0
for i in range(start_index, end_index):
    clean, hazy = haze_dataset[i]
    clean = restandardize_tensor(clean)
    hazy = restandardize_tensor(hazy)
    clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
    axes[row_index][0].imshow(clean_display)
    axes[row_index][0].set_title(f"Clean {i}")
    axes[row_index][0].axis("off")

    axes[row_index][1].imshow(hazy_display)
    axes[row_index][1].set_title(f"Hazy {i}")
    axes[row_index][1].axis("off")

    row_index += 1
# 6. Save the plot
save_path = "images/haze4k_hazy_clear_comparison.png"
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

# 7. Add the save command
print(f"Saving visualization to: {save_path}")
plt.savefig(
    save_path, dpi=300, bbox_inches="tight"
)  # Saves the figure with high resolution and tight bounds

# 8. Show the plot
plt.show()


# %%
## Loading the Validation Dataset of Haze4K
val_transform = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=256, split="test", verbose=True
)

haze_dataset = Haze4k_Dataset(
    root_dir=Path("dataset/haze4k"), split="val", transform=val_transform
)

N_COLS = 2
images_display = 4
N_ROWS = images_display
start_index = 10
end_index = start_index + images_display

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle(
    "Clear (Ground Truth) and Hazy Image Comparison (In the Haze4K Dataset)",
    fontsize=16,
)

row_index = 0
for i in range(start_index, end_index):
    clean, hazy = haze_dataset[i]
    clean = restandardize_tensor(clean)
    hazy = restandardize_tensor(hazy)
    clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
    axes[row_index][0].imshow(clean_display)
    axes[row_index][0].set_title(f"Clean {i}")
    axes[row_index][0].axis("off")

    axes[row_index][1].imshow(hazy_display)
    axes[row_index][1].set_title(f"Hazy {i}")
    axes[row_index][1].axis("off")

    row_index += 1
# 6. Save the plot
save_path = "images/haze4k_val_hazy_clear_comparison.png"
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

# 7. Add the save command
print(f"Saving visualization to: {save_path}")
plt.savefig(
    save_path, dpi=300, bbox_inches="tight"
)  # Saves the figure with high resolution and tight bounds

# 8. Show the plot
plt.show()


# %%
## Loading the O-HAZE Dataset
o_haze_path = Path("dataset/o-haze/O-HAZY/")
clear_path = o_haze_path / "GT"
haze_path = o_haze_path / "hazy"

clear_img_path = sorted(list(clear_path.glob("*.jpg")))
haze_img_path = sorted(list(haze_path.glob("*.jpg")))

print("Total Clear Images: ", len(clear_img_path))
print("Total Hazy Images: ", len(haze_img_path))


# %%
## Loading the O-HAZE Dataset with transforms
val_transform = get_haze_transforms(
    dataset_name="O-HAZE", resize_size=256, split="test", verbose=True
)

o_haze_path = Path("dataset/o-haze/O-HAZY/")
o_haze_dataset = OHAZE_Dataset(root_dir=o_haze_path, transform=val_transform)

N_COLS = 2
images_display = 3
N_ROWS = images_display
start_index = 10
end_index = start_index + images_display

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
fig.suptitle(
    "Clear (Ground Truth) and Hazy Image Comparison (In the O-HAZE Dataset)",
    fontsize=16,
)

row_index = 0
for i in range(start_index, end_index):
    clean, hazy = o_haze_dataset[i]
    clean = restandardize_tensor(clean)
    hazy = restandardize_tensor(hazy)
    clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
    axes[row_index][0].imshow(clean_display)
    axes[row_index][0].set_title(f"Clean {i}")
    axes[row_index][0].axis("off")

    axes[row_index][1].imshow(hazy_display)
    axes[row_index][1].set_title(f"Hazy {i}")
    axes[row_index][1].axis("off")

    row_index += 1
# 6. Save the plot
save_path = "images/o_haze_val_hazy_clear_comparison.png"
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

# 7. Add the save command
print(f"Saving visualization to: {save_path}")
plt.savefig(
    save_path, dpi=300, bbox_inches="tight"
)  # Saves the figure with high resolution and tight bounds

# 8. Show the plot
plt.show()


# %%
## Loading the Dense Haze dataset
densehaze_path = Path("dataset/dense-haze/")

train_transform = get_haze_transforms(
    dataset_name="DENSE-HAZE", resize_size=256, split="train", verbose=True
)
dense_haze = DENSE_Dataset(root_dir=densehaze_path, transform=train_transform)


def plotting_pair_images(dataset, num_instances=3, start_index=0, save_figure=False):
    N_COLS = 2
    N_ROWS = num_instances

    end_index = start_index + num_instances
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * ROWS))
    fig.suptitle(
        f"GT vs Haze Image Comparision in {dataset}",
        font_size=16,
    )
    row_index = 0
    for i in range(start_index, end_index):
        clean, hazy = dense_haze[i]
        clean = restandardize_tensor(clean)
        hazy = restandardize_tensor(hazy)
        clean_display, hazy_display = clean.permute(1, 2, 0), hazy.permute(1, 2, 0)
        axes[row_index][0].imshow(clean_display)
        axes[row_index][0].set_title(f"Clean {i}")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(hazy_display)
        axes[row_index][1].set_title(f"Hazy {i}")
        axes[row_index][1].axis("off")

        row_index += 1

    save_path = f"images/{dataset}_hazy_clear_comparison.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

    print(f"Saving visualization to: {save_path}")
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight"
    )  # Saves the figure with high resolution and tight bounds

    plt.show()


plotting_pair_images(dense_haze, save_figure=True)
# %%
