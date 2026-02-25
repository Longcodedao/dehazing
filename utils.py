import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
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
    DENSE_Haze_Dataset,
    NH_Haze_Dataset
)

from data.utils import get_haze_transforms, partition_dataset

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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        cudnn.deterministic = True
        cudnn.benchmark = False
        


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
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
            transform=train_transform,
        )
        val_dataset = RESIDE_SOTS_Indoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_SOTS_PATH),
            transform=val_transform,
            metadata="metadata_indoor.csv",
        )
        
    elif dataset_name == "RESIDE-OUTDOOR":
        if verbose: print(f"Loading RESIDE Outdoor (OTS)...")
        # IMPORTANT: Ensure your OTS subset file (e.g., dense_haze.txt) is used if needed
        # Modify the class init if you need to pass a specific .txt file list
        train_dataset = RESIDE_Outdoor(
            dataset_path=os.path.join(data_cfg.DATASET_ROOT, "outdoor-training-set"),
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
        
    elif dataset_name == "NHHAZE":
        if verbose: print(f"Loading NHHAZE...")
        train_dataset = NH_Haze_Dataset(
            root_dir=os.path.join(data_cfg.DATASET_ROOT, "nh-haze/NH-HAZE"),
            split="train",
            transform=train_transform,
        )
        val_dataset = NH_Haze_Dataset(
            root_dir=os.path.join(data_cfg.DATASET_ROOT, "nh-haze/NH-HAZE"),
            split="val",
            transform=val_transform,
        )
    elif dataset_name == "DENSEHAZE":
        if verbose: print("Loading DENSE-HAZE")
        densehaze_dataset = DENSE_Haze_Dataset(
            os.path.join(data_cfg.DATASET_ROOT, "dense-haze"),
        )
        train_dataset, val_dataset = partition_dataset(
            densehaze_dataset, 
            train_transform, 
            val_transform, 
            train_ratio = 0.91
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
    Creates a DataLoader for Evaluation (Validation/Test).
    Automatically handles Single-GPU and Multi-GPU (DDP) scenarios.
    
    Args:
        dataset_name: 'RESIDE-INDOOR', 'RESIDE-OUTDOOR', 'OHAZE', 'DENSEHAZE'
        dataset_root: Path to the main 'dataset' folder.
    """
    
    # 1. Setup Transforms (Normalize only, no resize)
    val_transform = get_haze_transforms(
        dataset_name=dataset_name, 
        resize_size=resolution, 
        split="val", 
        verbose=False
    )
    
    # 2. Select Dataset
    dataset = None
    name_upper = dataset_name.upper()

    if name_upper == "RESIDE-INDOOR":
        dataset = RESIDE_SOTS_Indoor(
            dataset_path=os.path.join(dataset_root, "reside-sots"),
            transform=val_transform,
            metadata="metadata_indoor.csv"
        )
    elif name_upper == "RESIDE-OUTDOOR":
        dataset = RESIDE_SOTS_Outdoor(
            dataset_path=os.path.join(dataset_root, "reside-sots"),
            transform=val_transform
        )
    elif name_upper == "OHAZE":
        dataset = OHAZE_Dataset(
            root_dir=os.path.join(dataset_root, "o-haze/O-HAZY"),
            transform=val_transform
        )
    elif name_upper == "DENSEHAZE":
        dataset = DENSE_Haze_Dataset(
            root_dir=os.path.join(dataset_root, "dense-haze"),
            transform=val_transform
        )
    elif name_upper == "NHHAZE":
        dataset = NH_Haze_Dataset(
            root_dir=os.path.join(dataset_root, "nh-haze/NH-HAZE"),
            transform=val_transform,
            split = "test"
        )
    else:
        raise ValueError(f"Unknown evaluation dataset: {dataset_name}")

    # 3. DDP Logic
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        # Splits data among GPUs so they don't evaluate the same images
        sampler = DistributedSampler(dataset, shuffle=False) 
    else:
        sampler = None

    # Only print on Rank 0 to avoid console spam
    if not is_distributed or (is_distributed and dist.get_rank() == 0):
        print(f"[{dataset_name}] Eval Dataset loaded: {len(dataset)} images. (DDP: {is_distributed})")

    # 4. Create Loader
    loader = DataLoader(
        dataset,
        batch_size=1,        # Always 1 for eval to handle different sizes
        shuffle=False,       # Never shuffle eval data
        sampler=sampler,     # Handles the DDP splitting
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



def predict_large_image(solver, full_img_tensor, device, progress=None, tile_size=256, overlap_ratio=0.25, batch_size=4, nfe=10):
    """
    Performs sliding-window inference with Rich progress bar integration.
    """
    b, c, h, w = full_img_tensor.shape
    
    # ... [Setup Canvas & Stride as before] ...
    output_canvas = torch.zeros((1, c, h, w), device=device)
    count_map = torch.zeros((1, 1, h, w), device=device)
    stride = int(tile_size * (1 - overlap_ratio))
    
    # ... [Setup Coordinates as before] ...
    h_starts = list(range(0, h - tile_size + stride, stride))
    w_starts = list(range(0, w - tile_size + stride, stride))
    if h_starts[-1] + tile_size > h: h_starts[-1] = h - tile_size
    if w_starts[-1] + tile_size > w: w_starts[-1] = w - tile_size
    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))

    # ... [Weight Mask as before] ...
    def get_weight_mask(size):
        coords = torch.linspace(0, 1, size, device=device)
        mask_1d = 1 - torch.abs(2 * coords - 1)
        mask_1d = mask_1d.unsqueeze(0)
        mask_2d = mask_1d.t() * mask_1d
        return mask_2d.unsqueeze(0).unsqueeze(0)

    weight_mask = get_weight_mask(tile_size)

    # --- RICH INTEGRATION START ---
    tiles = []
    coords = []
    all_patches = [(y, x) for y in h_starts for x in w_starts]
    
    # Create a temporary sub-task if progress is provided
    if progress is not None:
        # 'transient=True' means the bar disappears when finished
        tile_task_id = progress.add_task(f"  └─ Tiling ({len(all_patches)} patches)", total=len(all_patches), transient=True)
    
    for i, (y, x) in enumerate(all_patches):
        # Extract Crop
        crop = full_img_tensor[:, :, y:y+tile_size, x:x+tile_size].to(device)
        tiles.append(crop)
        coords.append((y, x))
        
        # Inference condition
        if len(tiles) == batch_size or i == len(all_patches) - 1:
            batch_tensor = torch.cat(tiles, dim=0)
            
            with torch.amp.autocast("cuda"):
                prediction_batch = solver.sample(batch_tensor, nfe=nfe)
            
            # Stitching
            for j, pred_tile in enumerate(prediction_batch):
                y_c, x_c = coords[j]
                pred_tile = pred_tile.unsqueeze(0)
                output_canvas[:, :, y_c:y_c+tile_size, x_c:x_c+tile_size] += pred_tile * weight_mask
                count_map[:, :, y_c:y_c+tile_size, x_c:x_c+tile_size] += weight_mask
            
            tiles = []
            coords = []
            
        # Update Rich Progress 
        if progress is not None:
            progress.update(tile_task_id, advance=1)
            
    # --- RICH INTEGRATION END ---

    final_output = output_canvas / (count_map + 1e-8)
    return final_output



@torch.no_grad()
def predict_large_image_vectorized(solver, full_img_tensor, device, progress=None, tile_size=256, overlap_ratio=0.25, batch_size=4, nfe=10):
    """
    Vectorized sliding window inference using F.unfold/F.fold.
    Much faster preparation than manual slicing loops.
    """
    b, c, h, w = full_img_tensor.shape
    
    # 1. Calculate Padding
    # Unfold drops pixels if they don't fit the stride. We pad to ensure coverage.
    stride = int(tile_size * (1 - overlap_ratio))
    
    # Calculate required height/width to be divisible by stride
    # Formula: (Size - Kernel) % Stride == 0
    pad_h = (stride - (h - tile_size) % stride) % stride
    pad_w = (stride - (w - tile_size) % stride) % stride
    
    # Add extra padding if the image is smaller than the tile
    if h < tile_size: pad_h += tile_size - h
    if w < tile_size: pad_w += tile_size - w

    # Pad image (Reflect padding usually best for dehazing to avoid borders)
    img_padded = F.pad(full_img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    hp, wp = img_padded.shape[2], img_padded.shape[3]

    # 2. Vectorized Unfold (Extract all patches at once)
    # Output shape: (1, C * tile_size * tile_size, Num_Patches)
    patches_raw = F.unfold(img_padded, kernel_size=tile_size, stride=stride)
    
    # Reshape to (Num_Patches, C, tile_size, tile_size) for the model
    # 1. Transpose -> (1, Num_Patches, Flattened_Patch)
    # 2. View -> (Num_Patches, C, tile_size, tile_size)
    num_patches = patches_raw.shape[2]
    patches_raw = patches_raw.transpose(1, 2).view(num_patches, c, tile_size, tile_size)
    
    # 3. Create Weight Mask (for smooth blending)
    # We create one mask and repeat it for all patches
    def get_weight_mask(size):
        coords = torch.linspace(0, 1, size, device=device)
        mask_1d = 1 - torch.abs(2 * coords - 1)
        mask_2d = mask_1d.unsqueeze(0).t() * mask_1d.unsqueeze(0)
        return mask_2d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    weight_patch = get_weight_mask(tile_size) # (1, 1, 256, 256)
    
    # 4. Batched Inference Loop
    # We can't process ALL patches at once (OOM), so we chunk them by 'batch_size'
    pred_patches_list = []
    
    # Rich Progress Setup
    task_id = None
    if progress is not None:
        task_id = progress.add_task(f"  └─ Vectorized ({num_patches} patches)", total=num_patches, transient=True)

    for i in range(0, num_patches, batch_size):
        # Select batch
        chunk = patches_raw[i : i + batch_size].to(device)
        
        with torch.amp.autocast("cuda"):
            # Model Output: (Batch, C, 256, 256)
            pred_chunk = solver.sample(chunk, nfe=nfe)
            
        # Apply weighting *immediately* to save memory later
        # (Batch, C, 256, 256) * (1, 1, 256, 256)
        pred_chunk_weighted = pred_chunk * weight_patch
        
        # Flatten back to (Batch, C*H*W) for folding
        pred_chunk_flat = pred_chunk_weighted.view(pred_chunk.shape[0], -1)
        pred_patches_list.append(pred_chunk_flat)
        
        if progress is not None:
            progress.update(task_id, advance=chunk.shape[0])

    # Concatenate all processed patches: (Num_Patches, Flattened_Pixels)
    pred_patches_all = torch.cat(pred_patches_list, dim=0)
    
    # 5. Fold (Stitch back together)
    # Reshape to (1, Flattened_Pixels, Num_Patches) for F.fold
    pred_patches_all = pred_patches_all.t().unsqueeze(0)
    
    # Fold creates the summed image
    output_sum = F.fold(
        pred_patches_all, 
        output_size=(hp, wp), 
        kernel_size=tile_size, 
        stride=stride
    )
    
    # 6. Normalize Weights (Account for overlaps)
    # We do the same "Unfold -> Fold" process for the weights to know what to divide by
    ones_patch = torch.ones(1, 1, tile_size, tile_size, device=device) * weight_patch
    ones_flat = ones_patch.view(1, -1).repeat(num_patches, 1).t().unsqueeze(0) # Repeat weight for every patch
    
    weight_sum = F.fold(
        ones_flat,
        output_size=(hp, wp),
        kernel_size=tile_size,
        stride=stride
    )
    
    # 7. Final Normalize & Crop
    final_img = output_sum / (weight_sum + 1e-8)
    
    # Crop back to original size (remove padding)
    final_img = final_img[:, :, :h, :w]
    
    return final_img
