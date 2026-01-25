import torch
import copy
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from typing import Union, Tuple
from torch.utils.data import Subset, Dataset

# --- Rich Imports ---
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

console = Console()

def print_transform_summary(
    name: str, geometric_sync: v2.Compose, haze_only: v2.Compose, common: v2.Compose
):
    """
    Prints a structured summary of the different transformation components using Rich.
    """
    # Create the root tree
    tree = Tree(f"[bold cyan]{name}[/]")

    # 1. Geometric Branch
    geo_branch = tree.add("[bold magenta]1. Geometric (Synchronous)[/]")
    geo_branch.add("[dim]Applies identically to CLEAR & HAZY for alignment[/]")
    geo_branch.add(str(geometric_sync))

    # 2. Appearance Branch
    haze_branch = tree.add("[bold yellow]2. Appearance (Hazy-Only)[/]")
    haze_branch.add("[dim]Simulates real-world haze variations[/]")
    haze_branch.add(str(haze_only))

    # 3. Common Branch
    common_branch = tree.add("[bold green]3. Common (Tensor & Norm)[/]")
    common_branch.add("[dim]Final prep: ToTensor, Normalize[/]")
    common_branch.add(str(common))

    # Print in a nice panel
    console.print(Panel(tree, title="[bold]Augmentation Pipeline[/]", expand=False, border_style="blue"))


def restandardize_tensor(
    tensor: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float]] = [0.5, 0.5, 0.5],
    std: Union[torch.Tensor, Tuple[float, float, float]] = [0.5, 0.5, 0.5],
) -> torch.Tensor:
    """
    Reverses normalization (z-score) -> (Tensor * STD) + MEAN.
    Returns tensor clipped to [0, 1].
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    de_normalized_tensor = (tensor * std) + mean
    final_tensor = torch.clamp(de_normalized_tensor, 0.0, 1.0)
    return final_tensor


def get_haze_transforms(
    dataset_name: str,
    resize_size: int = 640,
    split: str = "train",
    verbose: bool = False,
):
    """
    Defines PyTorch v2 transformations.
    
    Training: Resizes and augments.
    Validation: DOES NOT RESIZE (keeps original resolution).
    """

    # --- 1. Define Common Blocks ---
    
    # Train: Resize + Normalize
    train_common = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Val/Test: NO RESIZE (Original Size) + Normalize
    eval_common = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    # --- 2. SANITY CHECK LOGIC (New!) ---
    if split == "sanity":
        # STRICTLY DETERMINISTIC. 
        # No RandomCrop. No Flips. No Jitter. Just Resize & Normalize.
        def sanity_transform(clear_img, hazy_img):
            clean_img = train_common(clear_img)
            hazy_img = train_common(hazy_img)
            return clean_img, hazy_img
            
        if verbose:
            print("Transform Mode: SANITY (Deterministic Resize)")
        return sanity_transform
        
    # --- 3. Training Logic ---
    if split == "train":
        geometric_sync_transforms = v2.Compose([
            v2.RandomCrop(resize_size, pad_if_needed=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        # Dataset-specific augmentation logic
        # --- REVISED SAFER TRANSFORMATIONS ---
        # We reduce the intensity significantly.
        # The goal is "Domain Randomization" (robustness), not "Data Distortion".
        
        if dataset_name == "OHAZE":
            haze_only_transforms = v2.Compose([
                v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0),
            ])
        elif dataset_name == "DENSEHAZE":
            haze_only_transforms = v2.Compose([
                v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0),
                v2.RandomGrayscale(p=0.2),
            ])
        elif dataset_name == "NHHAZE":
            # NH-HAZE is non-homogeneous and very small (~55 pairs).
            # We need slightly more aggressive augmentation to prevent overfitting,
            # but we must be careful not to destroy the 'patchy' haze structure.
            haze_only_transforms = v2.Compose([
                v2.ColorJitter(
                    brightness=0.1,  # Moderate brightness changes
                    contrast=0.1,    # Moderate contrast
                    saturation=0.1,  # Haze affects saturation significantly
                    hue=0.0          # Keep hue 0 to preserve realistic outdoor colors
                ),
                # Grayscale helps the model focus on structure/texture rather 
                # than memorizing the specific color cast of the few training images.
                v2.RandomGrayscale(p=0.15),
            ])
            
        elif dataset_name in ["RESIDE-INDOOR", "HAZE4K"]:
            haze_only_transforms = v2.Compose([
                v2.ColorJitter(
                    brightness=0.15, 
                    contrast=0.15, 
                    saturation=0.15, 
                    hue=0.01
                )
            ])

        # Inrease the Saturation and Hue Jitter to force the model to
        # generalize to different weather conditions/times of day 
        elif dataset_name == "RESIDE-OUTDOOR":
            haze_only_transforms = v2.Compose([
                v2.ColorJitter(
                    brightness = 0.2, 
                    contrast = 0.2,
                    saturation = 0.2, # Stronger saturation jitter
                    hue = 0.05        # Allow slight color shifting (simulates time-of-day)
                ), 
                # Optional: Occasional Grayscale forces reliance on structure, not just color
                v2.RandomGrayscale(p=0.1),
            ])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        
        def haze_transform(clear_img, hazy_img):
            # Apply geometric (sync), appearance (hazy only), then common (resize+norm)
            clean_img, hazy_img = geometric_sync_transforms(clear_img, hazy_img)
            hazy_img = haze_only_transforms(hazy_img)
            clean_img = train_common(clean_img)
            hazy_img = train_common(hazy_img)
            return clean_img, hazy_img

        if verbose:
            print_transform_summary(
                f"{dataset_name} | TRAIN | {resize_size}x{resize_size}",
                geometric_sync_transforms,
                haze_only_transforms,
                train_common,
            )
        return haze_transform

    # --- 3. Validation Logic ---
    else:
        def val_transform(clear_img, hazy_img):
            # Only apply normalization, NO RESIZING
            clean_img = eval_common(clear_img)
            hazy_img = eval_common(hazy_img)
            return clean_img, hazy_img

        if verbose:
            print_transform_summary(
                f"{dataset_name} | VAL/TEST | Original Size",
                v2.Identity(),
                v2.Identity(),
                eval_common,
            )
        return val_transform


def partition_dataset(dataset, train_transform, val_transform, 
                      train_ratio=0.8, seed = 42):
    indices = torch.randperm(len(dataset)).tolist()
    num_train = int(len(dataset) * train_ratio)
    
    train_subset = Subset(copy.deepcopy(dataset), indices[:num_train])
    val_subset = Subset(copy.deepcopy(dataset), indices[num_train:])

    # Inject transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    return train_subset, val_subset


def plotting_pair_images(dataset, num_instances=3, start_index=0, save_figure=False):
    N_COLS = 2
    N_ROWS = num_instances
    end_index = start_index + num_instances
    
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(8, 5 * N_ROWS))
    fig.suptitle(f"GT vs Haze Comparison", fontsize=16)

    row_index = 0
    for i in range(start_index, end_index):
        clean, hazy = dataset[i]
        clean = restandardize_tensor(clean)
        hazy = restandardize_tensor(hazy)
        
        axes[row_index][0].imshow(clean.permute(1, 2, 0))
        axes[row_index][0].set_title(f"Clean {i}")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(hazy.permute(1, 2, 0))
        axes[row_index][1].set_title(f"Hazy {i}")
        axes[row_index][1].axis("off")
        row_index += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_figure:
        path = "hazy_clear_comparison.png"
        console.print(f"[bold green]Saving visualization to: {path}[/]")
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()