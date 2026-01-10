import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_ddp import setup_ddp, clean_ddp, is_main_process, reduce_tensor
import yaml
import warnings

# -- CONFIG IMPORTS --
from yacs.config import CfgNode as CN
from config import get_cfg_defaults

# -- DATASET IMPORTS --
# (Assuming your data module exports these)
from data import RESIDE_Indoor, Haze4k_Dataset, OHAZE_Dataset, DENSE_Haze_Dataset
from utils import (
    convert_cfg_to_dict,
    set_seed,
    get_loaders_for_stage, # Ensure this handles resizing!
)

# -- MODEL & TRAINER IMPORTS --
from model import FM_PhysMamba_UNET
from losses import FM_PhysicalLoss
from trainer import DehazeTrainer

# -- RICH IMPORTS --
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore")

# --- 1. ARGUMENTS ---
def create_args():
    parser = argparse.ArgumentParser(description="Dehaze Flow Matching Trainer")
    
    # Config
    parser.add_argument("--pretrain-config", default="configs/train_cfgs/pretrain_schedule.yaml", type=str)
    parser.add_argument("--model-config", default="small", type=str, help="Path to model config or 'small'/'large'")

    # System Overrides
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--pin_memory", type=int, choices=[0, 1], default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    
    # Data Overrides
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="RESIDE", help="Name of dataset (RESIDE, HAZE4K)")
    parser.add_argument("--resume", type=str, default="")
    
    # Generic Overrides
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def setup_config(args):
    # 1. Get Defaults
    cfg = get_cfg_defaults()
    
    # 2. Merge Schedule YAML
    if args.pretrain_config and os.path.exists(args.pretrain_config):
        cfg.merge_from_file(args.pretrain_config)
    
    # 3. Apply Overrides
    if args.num_workers is not None: cfg.NUM_WORKERS = args.num_workers
    if args.seed is not None: cfg.SEED = args.seed
    if args.log_dir is not None: cfg.LOG_DIR = args.log_dir
    if args.pin_memory is not None: cfg.PIN_MEMORY = bool(args.pin_memory)
    if args.checkpoint_dir is not None: cfg.CHECKPOINT_DIR = args.checkpoint_dir
    if args.dataset_root is not None: cfg.DATA.DATASET_ROOT = args.dataset_root
    if args.resume: cfg.TRAIN.RESUME_PATH = args.resume
    if args.opts: cfg.merge_from_list(args.opts)
        
    cfg.freeze()
    return cfg
    

if __name__ == "__main__":
    local_rank = setup_ddp()
    console = Console() if is_main_process() else None

    # 1. Setup
    args = create_args()
    cfg = setup_config(args)
    
    # 2. Reproducibility
    seed = (cfg.SEED if hasattr(cfg, 'SEED') else 42) + local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if is_main_process():
        console.print(Panel(f"[bold green]Dehaze Flow Matching[/]\n[yellow]Model: {args.model_config}[/]", expand=False))
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # 3. Model & Loss
    # We pass the CLI argument for model config ("small", "large", or path)
    model = FM_PhysMamba_UNET(model_cfg_path=args.model_config)
    
    # Pass cfg to loss so it can read weights (W_FLOW, W_PHYS, etc.)
    criterion = FM_PhysicalLoss(cfg) 
    
    # 4. Trainer
    trainer = DehazeTrainer(cfg, model, criterion, local_rank)

    # 5. Log Config to TensorBoard
    if is_main_process() and trainer.writer is not None:
        cfg_str = yaml.dump(convert_cfg_to_dict(cfg), sort_keys=False)
        trainer.writer.add_text("Configuration", f"```yaml\n{cfg_str}\n```", 0)

    # 6. Resume if needed
    if cfg.TRAIN.RESUME_PATH:
        trainer.load_checkpoint(cfg.TRAIN.RESUME_PATH)

    # ---------------------------------------------------------
    # PROGRESSIVE TRAINING LOOP
    # ---------------------------------------------------------
    
    # Use schedule from config, or fallback to single stage
    schedule = cfg.SCHEDULE if len(cfg.SCHEDULE) > 0 else [
        {
            "RESOLUTION": 256, 
            "EPOCHS": cfg.TRAIN.EPOCHS,
            "BATCH_SIZE": cfg.TRAIN.BATCH_SIZE,
            "EVAL_INTERVAL": cfg.EVAL.EVAL_INTERVAL
        }
    ]

    cumulative_target_epoch = 0
    
    for stage_idx, stage_cfg in enumerate(schedule):
        # Handle access for both Dict (YAML) and CfgNode
        res = stage_cfg.get('RESOLUTION', 256)
        stage_epochs = stage_cfg.get('EPOCHS', 100)
        batch_size = stage_cfg.get('BATCH_SIZE', 16)
        
        # Calculate when this stage should end
        cumulative_target_epoch += stage_epochs

        # Skip if we resumed past this stage
        if trainer.epoch > cumulative_target_epoch:
            if is_main_process():
                console.print(f"[dim]Skipping Stage {stage_idx+1} (Res {res}) - Already completed.[/]")
            continue

        if is_main_process():
            console.print(Panel(
                f"[bold cyan]Starting Stage {stage_idx+1}/{len(schedule)}[/]\n"
                f"Resolution: {res}x{res}\n"
                f"Batch Size: {batch_size}\n"
                f"Stage Duration: {stage_epochs} Epochs\n"
                f"Target Epoch: {cumulative_target_epoch}",
                title="Progressive Schedule"
            ))

        # --- A. Re-Initialize DataLoaders ---
        train_loader, val_loader = get_loaders_for_stage(
            cfg, 
            args.dataset_name,
            resolution=res, 
            batch_size=batch_size, 
            rank=local_rank
        )

        # --- B. Train ---
        # trainer.fit runs from current epoch -> cumulative_target_epoch
        trainer.fit(
            train_loader, 
            val_loader, 
            max_epochs=cumulative_target_epoch, 
            save_dir=os.path.join(cfg.LOG_DIR, f"stage_{stage_idx}_res{res}")
        )
        
        # --- C. Checkpoint Stage ---
        trainer.save_checkpoint(
            os.path.join(cfg.CHECKPOINT_DIR, f"stage_{stage_idx}_finished.pt")
        )

    clean_ddp()