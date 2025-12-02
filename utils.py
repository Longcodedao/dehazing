import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.utils import get_haze_transforms
from data import RESIDE_Indoor, RESIDE_SOTS_Indoor
from yacs.config import CfgNode as CN
import os


def convert_cfg_to_dict(cfg_node):
    """
    Recursively converts a YACS CfgNode (and lists of CfgNodes)
    into standard Python dictionaries and lists.
    """
    if not isinstance(cfg_node, CN):
        # If it's a list, we need to check if items inside are CfgNodes
        if isinstance(cfg_node, list):
            return [convert_cfg_to_dict(item) for item in cfg_node]
        return cfg_node
    else:
        # Convert CfgNode to dict
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v)
        return cfg_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders_for_stage(cfg, resolution, batch_size, verbose=False):
    ## Training RESIDE Indoor
    data_cfg = cfg.DATA

    train_transform_reside = get_haze_transforms(
        dataset_name="RESIDE",
        resize_size=resolution,
        split="train",
        verbose=verbose,
    )

    val_transform_reside = get_haze_transforms(
        dataset_name="RESIDE",
        resize_size=resolution,
        split="val",
        verbose=verbose,
    )
    train_dataset = RESIDE_Indoor(
        dataset_path=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
        transform=train_transform_reside,
    )
    val_dataset = RESIDE_SOTS_Indoor(
        dataset_path="dataset/reside-sots/",
        transform=val_transform_reside,
        metadata="metadata_indoor.csv",
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.EVAL.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, train_sampler


# Toggle the gradients
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad


## Pad the images for evaluation
def pad_to_multiple(image_tensor, multiple=16):
    """
    Pads the height (H) and width (W) of the image_tensor (B, C, H, W)
    to be a multiple of the specified factor.
    """
    b, c, h, w = image_tensor.shape

    # Calculate required padded dimensions
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    # Apply padding only to the bottom and right
    # (padding_left, padding_right, padding_top, padding_bottom)
    padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return padded_tensor, pad_h, pad_w


## Unpad the image
def unpad(padded_tensor, pad_h, pad_w):
    """
    Crops the padded tensor back to the original size.
    """
    if pad_h == 0 and pad_w == 0:
        return padded_tensor

    h_padded = padded_tensor.shape[2]
    w_padded = padded_tensor.shape[3]

    # Crop from (0, 0) up to (h_padded - pad_h, w_padded - pad_w)
    return padded_tensor[:, :, : h_padded - pad_h, : w_padded - pad_w]
