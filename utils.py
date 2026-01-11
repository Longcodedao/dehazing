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
from data import RESIDE_Indoor, RESIDE_SOTS_Indoor, Haze4k_Dataset
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
        dataset_name (str): Name of the dataset (e.g., 'RESIDE', 'HAZE4K').
        resolution: Target size for TRAIN images (e.g., 256).
        batch_size: Batch size for TRAIN images.
        rank: Process rank (for printing verbose info only on rank 0).
    """
    verbose = (rank == 0)
    data_cfg = cfg.DATA
    
    # Check if Distributed Processing is Initialized
    is_distributed = dist.is_available() and dist.is_initialized()

    # 1. Train Transform (Resizes to 'resolution')
    train_transform = get_haze_transforms(
        dataset_name="RESIDE",  # Or pass dataset_name if logic differs per dataset
        resize_size=resolution,
        split="train",
        verbose=verbose, 
    )

    # 2. Val Transform (Keeps ORIGINAL size, ignores 'resolution')
    val_transform = get_haze_transforms(
        dataset_name="RESIDE",
        resize_size=resolution, # Passed but ignored inside the function for 'val'
        split="val",
        verbose=verbose,
    )

    # --- 3. Instantiate Datasets Dynamically ---
    if dataset_name == "RESIDE":
        # RESIDE has separate classes for Train (ITS/OTS) and Val (SOTS)
        train_dataset = RESIDE_Indoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
            transform=train_transform,
        )
        val_dataset = RESIDE_SOTS_Indoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "reside-sots"),
            transform=val_transform,
            metadata="metadata_indoor.csv",
        )
        
    elif dataset_name == "HAZE4K":
        # Haze4k usually splits a single dataset folder
        train_dataset = Haze4k_Dataset(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "Haze4k"),
            split="train",
            transform=train_transform,
        )
        val_dataset = Haze4k_Dataset(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "Haze4k"),
            split="val", # or 'test'
            transform=val_transform,
        )
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in get_loaders_for_stage")
        
    # --- 4. Samplers (Hybrid Logic) ---
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False # Sampler handles shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True # Loader handles shuffle

    # --- 5. Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train, 
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    
    # Validation Loader: Must use batch_size=1 if preserving original varying sizes
    val_loader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader
    

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