import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from data.utils import get_haze_transforms, partition_dataset
from data import RESIDE_Indoor, Haze4k_Dataset
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
    reside_dataset = RESIDE_Indoor(
        dataset_path=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
        transform=None,
    )
    train_reside_dataset, val_reside_dataset = partition_dataset(
        reside_dataset,
        train_transform_reside,
        val_transform_reside,
        train_ratio=data_cfg.TRAIN_RATIO,
    )

    # Loading the Haze4k Dataset
    train_transform_haze4k = get_haze_transforms(
        dataset_name="HAZE4K", resize_size=resolution, split="train", verbose=verbose
    )
    val_transform_haze4k = get_haze_transforms(
        dataset_name="HAZE4K", resize_size=resolution, split="val", verbose=verbose
    )
    haze_4k_train = Haze4k_Dataset(
        root_dir=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
        split="train",
        transform=train_transform_haze4k,
    )
    haze_4k_val = Haze4k_Dataset(
        root_dir=os.path.join(data_cfg.DATASET_ROOT, data_cfg.RESIDE_INDOOR_PATH),
        split="val",
        transform=val_transform_haze4k,
    )
    train_dataset = ConcatDataset([train_reside_dataset, haze_4k_train])
    val_dataset = ConcatDataset([val_reside_dataset, haze_4k_val])

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
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, train_sampler
