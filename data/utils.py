import torch
from torchvision import transforms as v2
import math


def print_transform_summary(
    name: str, geometric_sync: v2.Compose, haze_only: v2.Compose, common: v2.Compose
):
    """Prints a structured summary of the different transformation components."""

    print(f"## 📝 Dehazing Transformations Summary: {name}")
    print("---")

    # 1. Geometric Transforms (Applied Synchronously to Clear and Hazy)
    print("### 1. Geometric (Synchronous) Transforms:")
    print("Applies identically to both CLEAR and HAZY images for pixel alignment.")
    print(geometric_sync)

    # 2. Hazy-Only Transforms (Applied Asynchronously to Hazy)
    print("\n### 2. Appearance (Hazy-Only) Transforms:")
    print("Applied only to the HAZY image to simulate real-world haze variations.")
    print(haze_only)

    # 3. Common Transforms (Applied to both before model input)
    print("\n### 3. Common (Tensor Conversion & Normalization) Transforms:")
    print("Applied to both images before feeding to the model.")
    print(common)


# --------------------------------------------------------------------------


def get_reside_haze_transforms(
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
            v2.Resize(resize_size, antialias=True),
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
                name=f"Training Split (Size: {resize_size}x{resize_size})",
                geometric_sync=geometric_sync_transforms,
                haze_only=haze_only_transforms,
                common=common_transforms,
            )

        return haze_transform

    else:  # split == "val" or "test"
        # --- Handle Validation/Test Split ---
        if verbose:
            print_transform_summary(
                name=f"Validation/Test Split (Size: {resize_size}x{resize_size})",
                geometric_sync=v2.Compose([]),  # No random geometric transforms
                hazy_only=v2.Compose([]),  # No color jitter/grayscale
                common=common_transforms,
            )
        return (
            common_transforms  # Returns the Compose object for single-image application
        )
        # Note: In this case, your Dataset.__getitem__ would need
        # to apply it to both images separately.
