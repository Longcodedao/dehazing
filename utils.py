import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from yacs.config import CfgNode as CN
import os

# Import your datasets and transforms
# (Ensure these imports match your project structure)
from data import (
    RESIDE_Indoor, 
    RESIDE_Outdoor,
    RESIDE_SOTS_Indoor, 
    RESIDE_SOTS_Outdoor,
    Haze4k_Dataset, 
    OHAZE_Dataset,
    DENSE_Haze_Dataset
)

from data.utils import get_haze_transforms

def convert_cfg_to_dict(cfg_node):
    """Recursively converts a YACS CfgNode to a standard Python dict."""
    if not isinstance(cfg_node, CN):
        if isinstance(cfg_node, list):
            return [convert_cfg_to_dict(item) for item in cfg_node]
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v)
        return cfg_dict

def set_seed(seed):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders_for_stage(cfg, dataset_name, resolution, batch_size, rank=0):
    """
    Creates DataLoaders for a specific training stage.
    Automatically handles Single-GPU vs Distributed (DDP) logic.
    
    Args:
        dataset_name (str): 'RESIDE-INDOOR', 'RESIDE-OUTDOOR', 'HAZE4K', etc.
        resolution: Target size for TRAIN images (e.g., 256).
        batch_size: Batch size for TRAIN images.
        rank: Process rank (for printing verbose info only on rank 0).
    """
    verbose = (rank == 0)
    data_cfg = cfg.DATA
    
    # Check if Distributed Processing is Initialized
    is_distributed = dist.is_available() and dist.is_initialized()

    # --- 1. Define Transforms ---
    # Train: Resize + Augment
    train_transform = get_haze_transforms(
        dataset_name=dataset_name, 
        resize_size=resolution,
        split="train",
        verbose=verbose, 
    )

    # Val: Keep Original Size + Normalize
    val_transform = get_haze_transforms(
        dataset_name=dataset_name,
        resize_size=resolution, # Ignored for Val, but passed for API consistency
        split="val",
        verbose=verbose,
    )

    # --- 2. Instantiate Datasets Dynamically ---
    train_dataset = None
    val_dataset = None

    if dataset_name == "RESIDE-INDOOR":
        if verbose: print(f"Loading RESIDE Indoor (ITS)...")
        train_dataset = RESIDE_Indoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "reside-indoor"),
            transform=train_transform,
        )
        val_dataset = RESIDE_SOTS_Indoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "reside-sots"),
            transform=val_transform,
            metadata="metadata_indoor.csv",
        )
        
    elif dataset_name == "RESIDE-OUTDOOR":
        if verbose: print(f"Loading RESIDE Outdoor (OTS)...")
        # IMPORTANT: Ensure your OTS subset file (e.g., dense_haze.txt) is used if needed
        # Modify the class init if you need to pass a specific .txt file list
        train_dataset = RESIDE_Outdoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "reside-outdoor"),
            transform=train_transform,
        )
        val_dataset = RESIDE_SOTS_Outdoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "reside-sots"),
            transform=val_transform,
        )

    elif dataset_name == "HAZE4K":
        if verbose: print(f"Loading Haze4k...")
        train_dataset = Haze4k_Dataset(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "Haze4k"),
            split="train",
            transform=train_transform,
        )
        val_dataset = Haze4k_Dataset(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "Haze4k"),
            split="val",
            transform=val_transform,
        )
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in get_loaders_for_stage")
        
    # --- 3. Samplers (Hybrid Logic) ---
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False # Sampler handles shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True # Loader handles shuffle

    # --- 4. Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train, 
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    
    # Validation Loader: Must use batch_size=1 to handle varying image sizes
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    if verbose:
        print(f"Data Loaders Ready. Train: {len(train_loader)} batches, Val: {len(val_loader)} images.")

    return train_loader, val_loader


# --- 2. The Evaluation Loader Factory ---
def get_eval_loader(
    dataset_name: str,
    dataset_root: str,
    resolution: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Creates a DataLoader strictly for Evaluation (Validation/Test).
    
    Features:
    - Batch Size = 1 (Required for metrics on varying resolution images).
    - No Shuffling (Consistent evaluation order).
    - 'val' Transforms (No resizing, only normalization).
    
    Args:
        dataset_name: 'RESIDE-INDOOR', 'RESIDE-OUTDOOR', 'OHAZE', 'DENSEHAZE'
        dataset_root: Path to the main 'dataset' folder.
    """
    
    # 1. Get Transform (Normalize only, no resize)
    # We pass 'resolution' just to satisfy the function sig, but split='val' ignores it.
    val_transform = get_haze_transforms(
        dataset_name=dataset_name, 
        resize_size=resolution, 
        split="val", 
        verbose=False
    )
    
    dataset = None
    
    # --- RESIDE SOTS INDOOR ---
    if dataset_name.upper() == "RESIDE-INDOOR":
        dataset = RESIDE_SOTS_Indoor(
            dataset_path=os.path.join(dataset_root, "reside-sots"),
            transform=val_transform,
            metadata="metadata_indoor.csv" # Ensure this CSV exists in reside-sots
        )
        
    # --- RESIDE SOTS OUTDOOR ---
    elif dataset_name.upper() == "RESIDE-OUTDOOR":
        dataset = RESIDE_SOTS_Outdoor(
            dataset_path=os.path.join(dataset_root, "reside-sots"),
            transform=val_transform
        )

    # --- O-HAZE ---
    elif dataset_name.upper() == "OHAZE":
        # Structure: dataset/O-Haze/hazy, dataset/O-Haze/GT
        dataset = OHAZE_Dataset(
            root_dir=os.path.join(dataset_root, "o-haze"),
            transform=val_transform
        )

    # --- DENSE-HAZE ---
    elif dataset_name.upper() == "DENSEHAZE":
        # Structure: dataset/Dense-Haze/hazy, dataset/Dense-Haze/GT
        dataset = DENSE_Haze_Dataset(
            root_dir=os.path.join(dataset_root, "dense-Haze"),
            transform=val_transform
        )
        
    else:
        raise ValueError(f"Unknown evaluation dataset: {dataset_name}")

    print(f"[{dataset_name}] Eval Dataset loaded with {len(dataset)} images.")

    # 3. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,        # CRITICAL for evaluation
        shuffle=False,       # CRITICAL for consistent comparisons
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return loader

    

# --- Padding Utilities for Inference ---
# Use these inside your evaluation loop if you encounter size mismatches 
# with model architecture requirements (e.g. UNet needs div by 16)

def pad_to_multiple(image_tensor, multiple=16):
    b, c, h, w = image_tensor.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded_tensor, pad_h, pad_w


def unpad(padded_tensor, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return padded_tensor
    h_padded, w_padded = padded_tensor.shape[2], padded_tensor.shape[3]
    return padded_tensor[:, :, : h_padded - pad_h, : w_padded - pad_w]