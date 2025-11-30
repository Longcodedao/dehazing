import torch
from torchvision.transforms import v2
from typing import Union, Tuple
import matplotlib.pyplot as plt
import copy
from torch.utils.data import Subset, Dataset


def print_transform_summary(
    name: str, geometric_sync: v2.Compose, haze_only: v2.Compose, common: v2.Compose
):
    """Prints a structured summary of the different transformation components."""

    print(f"## 📝 Dehazing Transformations Summary: {name}")
    print("------------------------------------------------")

    # 1. Geometric Transforms (Applied Synchronously to Clear and Hazy)
    print("1. Geometric (Synchronous) Transforms:")
    print("Applies identically to both CLEAR and HAZY images for pixel alignment.")
    print(geometric_sync)

    # 2. Hazy-Only Transforms (Applied Asynchronously to Hazy)
    print("\n2. Appearance (Hazy-Only) Transforms:")
    print("Applied only to the HAZY image to simulate real-world haze variations.")
    print(haze_only)

    # 3. Common Transforms (Applied to both before model input)
    print("\n3. Common (Tensor Conversion & Normalization) Transforms:")
    print("Applied to both images before feeding to the model.")
    print(common)
    print("\n")


def restandardize_tensor(
    tensor: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float]] = [0.5, 0.5, 0.5],
    std: Union[torch.Tensor, Tuple[float, float, float]] = [0.5, 0.5, 0.5],
) -> torch.Tensor:
    """
    Reverses the normalization operation (z-score standardization) on an image tensor
    to prepare it for visualization.

    The reversal formula is: De-normalized Tensor = (Normalized Tensor * STD) + MEAN

    Args:
        tensor: The normalized image tensor (C, H, W) or (B, C, H, W).
        mean: The mean used during normalization.
        std: The standard deviation used during normalization.

    Returns:
        The de-normalized tensor, clipped to the range [0, 1].
    """
    # Ensure mean and std are tensors with the correct shape (C, 1, 1) for broadcasting
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
            -1, 1, 1
        )
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    # Handle batch dimension: unsqueeze mean/std if tensor is (B, C, H, W)
    if tensor.dim() == 4:
        # Add a batch dimension to mean/std for broadcasting across the batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    # 1. Reverse the Normalization (Multiply by STD, Add MEAN)
    de_normalized_tensor = (tensor * std) + mean

    # 2. Clamp the values to the valid [0, 1] range
    # Necessary because model predictions might fall outside of [-1, 1] after de-normalization.
    final_tensor = torch.clamp(de_normalized_tensor, 0.0, 1.0)

    return final_tensor


# --------------------------------------------------------------------------
def get_haze_transforms(
    dataset_name: str,
    resize_size: int = 640,
    split: str = "train",
    verbose: bool = False,
):
    """
    Defines the PyTorch vision transformations for dehazing.
    (The Augmentations may differ according to each dataset)

    Args:
        dataset_name (str): The name of the dataset
        resize_size (int): The target size for images (H x W).
        split (str): 'train' for augmentations, 'test'/'val' for standardization.
        verbose (bool): If True, prints a structured summary of the applied transforms.

    Returns:
        function: A function that takes (clear_img, hazy_img) and returns (clear_tensor, hazy_tensor).
    """

    # --- Component Definitions ---
    common_transforms = v2.Compose(
        [
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    if split == "train":
        geometric_sync_transforms = v2.Compose(
            [
                v2.RandomCrop(resize_size, pad_if_needed=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

        # Real-world data: Minimal color jitter to avoid generating unrealistic haze
        if dataset_name == "O-HAZE":
            haze_only_transforms = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                    ),
                ]
            )
        elif dataset_name == "DENSE-HAZE":
            # Synthetic, dense haze: High jitter and more grayscale to focus on structure
            # Use the luminance to reconstruct that instead of using all RGB features
            # When we have the saturation loss
            haze_only_transforms = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15
                    ),
                    v2.RandomGrayscale(p=0.2),
                ]
            )
        elif dataset_name in ["RESIDE", "HAZE4K"]:
            haze_only_transforms = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    v2.RandomGrayscale(p=0.1),
                ]
            )
        else:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Please use 'RESIDE', 'HAZE4K', 'O-HAZE', or 'DENSE-HAZE'."
            )

        # --- The Core Transformation Function ---
        def haze_transform(clear_img, hazy_img):
            # 1. Apply Geometric Transforms (Synchronous)
            clean_img, hazy_img = geometric_sync_transforms(clear_img, hazy_img)

            # 2. Apply Hazy-Only Transforms (Asynchronously)
            hazy_img = haze_only_transforms(hazy_img)

            # 3. Apply Common Transforms
            clean_img = common_transforms(clean_img)
            hazy_img = common_transforms(hazy_img)

            return clean_img, hazy_img

        # --- Print Summary (Conditional) ---
        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tTraining Split (Size: {resize_size}x{resize_size})",
                geometric_sync=geometric_sync_transforms,
                haze_only=haze_only_transforms,
                common=common_transforms,
            )

        return haze_transform

    else:  # split == "val" or "test"
        # --- Handle Validation/Test Split ---
        def val_transform(clear_img, hazy_img):
            clean_img = common_transforms(clear_img)
            hazy_img = common_transforms(hazy_img)
            return clean_img, hazy_img

        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tValidation/Test Split (Size: {resize_size}x{resize_size})",
                geometric_sync=v2.Identity(),  # No random geometric transforms
                haze_only=v2.Identity(),  # No color jitter/grayscale
                common=common_transforms,
            )
        return val_transform


def partition_dataset(
    dataset: Dataset,
    train_transform: callable,
    val_transform: callable,
    train_ratio=0.8,
):
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(len(dataset) * train_ratio)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset_base = copy.deepcopy(dataset)
    val_dataset_base = copy.deepcopy(dataset)

    train_dataset_base.transform = train_transform
    val_dataset_base.transform = val_transform

    train_subset = Subset(train_dataset_base, train_indices)
    val_subset = Subset(val_dataset_base, val_indices)

    return train_subset, val_subset


def plotting_pair_images(dataset, num_instances=3, start_index=0, save_figure=False):
    N_COLS = 2
    N_ROWS = num_instances

    end_index = start_index + num_instances
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(4 * N_COLS, 5 * N_ROWS))
    fig.suptitle(
        f"GT vs Haze Image Comparision in {dataset}",
        fontsize=16,
    )
    row_index = 0
    for i in range(start_index, end_index):
        clean, hazy = dataset[i]
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

    if save_figure:
        print(f"Saving visualization to: {save_path}")
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight"
        )  # Saves the figure with high resolution and tight bounds

    plt.show()
