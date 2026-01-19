import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    get_loaders_for_stage, 
    get_eval_loader
)
from utils_ddp import setup_ddp, clean_ddp, is_main_process

# -- MODEL & TRAINER IMPORTS --
from model import FM_PhysMamba_UNET
from losses import FM_PhysicalLoss
from trainer import DehazeTrainer

# -- RICH IMPORTS --
from rich.console import Console
from rich.panel import Panel

warnings.filterwarnings("ignore")

def is_dist_avail_and_initialized():
    """Checks if the script was launched with torchrun/distributed vars."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

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

    # Fine-tuning options
    parser.add_argument("--pretrained_model", type=str, default="", 
                        help="Path to load ONLY model weights (for fine-tuning)")
    parser.add_argument("--strict_load", action="store_true", 
                        help="Whether to load the state_dict strictly")
    
    # Data Overrides
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="RESIDE", help="Name of dataset (RESIDE, HAZE4K)")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    
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
    # --- 1. Conditional DDP Setup ---
    # We check if environment variables for DDP exist.
    if is_dist_avail_and_initialized():
        local_rank = setup_ddp()
        distributed = True
    else:
        # Single GPU fallback
        local_rank = 0
        distributed = False
        # Optional: Set specific device if needed, otherwise Trainer handles it
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    console = Console() if is_main_process() else None

    # --- 2. Setup Args & Config ---
    args = create_args()
    cfg = setup_config(args)

    # =========================================================
    # [CRITICAL UPDATE] Calculate Total Epochs from Schedule
    # =========================================================
    if len(cfg.SCHEDULE) > 0:
        total_schedule_epochs = sum([s.get('EPOCHS', 0) for s in cfg.SCHEDULE])
        
        # Override the global epoch count so the Scheduler knows 
        # it has 150 epochs to decay, not just 100.
        # We need to unfreeze, modify, and refreeze.
        cfg.defrost()
        cfg.TRAIN.EPOCHS = total_schedule_epochs
        cfg.freeze()
        
        if is_main_process():
            console.print(f"[bold yellow]Progressive Schedule Detected:[/]")
            console.print(f"Total Epochs set to: [bold cyan]{total_schedule_epochs}[/] (Sum of stages)")
    
    # --- 3. Reproducibility ---
    # We add local_rank to seed to ensure different seeds on different GPUs
    seed = (cfg.SEED if hasattr(cfg, 'SEED') else 42) + local_rank
    set_seed(seed) # Assuming set_seed handles torch, cuda, and random

    if is_main_process():
        mode_str = "Distributed (DDP)" if distributed else "Single GPU"
        console.print(Panel(
            f"[bold green]Dehaze Flow Matching[/]\n"
            f"[yellow]Mode:[/yellow] {mode_str}\n"
            f"[yellow]Model:[/yellow] {args.model_config}\n"
            f"[yellow]Dataset:[/yellow] {args.dataset_name}", 
            expand=False, title="Initialization"
        ))
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # --- 4. Model & Loss ---
    # We pass the CLI argument for model config ("small", "large", or path)
    model = FM_PhysMamba_UNET(
        model_cfg_path = args.model_config,
        gradient_checkpointing = args.gradient_checkpointing,
        use_version = 2
    )

    if args.pretrained_model:
        if is_main_process():
            console.print(f"[bold green]Loading pretrained weights from:[/][white] {args.pretrained_model}")
        # Load the file
        checkpoint = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)

    
        # Extract model state even if it's a full checkpoint [1]
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Clean DDP keys (removing 'module.') so it can load on any setup [1]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load into model
        msg = model.load_state_dict(state_dict, strict=args.strict_load)
        if is_main_process():
            console.print(f"[yellow]Load status:[/] {msg}")

    
    # Pass cfg to loss so it can read weights (W_FLOW, W_PHYS, etc.)
    criterion = FM_PhysicalLoss(cfg) 
    
    # --- 5. Trainer ---
    # The trainer class we refactored earlier handles the DDP wrapping internally
    trainer = DehazeTrainer(cfg, model, criterion, local_rank)
    
    # --- 6. Log Config to TensorBoard ---
    if is_main_process() and trainer.writer is not None:
        cfg_str = yaml.dump(convert_cfg_to_dict(cfg), sort_keys=False)
        trainer.writer.add_text("Configuration", f"```yaml\n{cfg_str}\n```", 0)

    
    # --- 7. Resume if needed ---
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

    # Variables to track the last state for final testing
    last_save_dir = None
    # Initialize with None; will be updated in loop
    val_loader = None

    # indoor_val_loader = get_eval_loader(
    #     dataset_name="RESIDE-INDOOR",
    #     dataset_root=args.dataset_root,
    #     num_workers=args.num_workers
    # )
    
    for stage_idx, stage_cfg in enumerate(schedule):
        # Handle access for both Dict (YAML) and CfgNode
        res = stage_cfg.get('RESOLUTION', 256)
        stage_epochs = stage_cfg.get('EPOCHS', 100)
        batch_size = stage_cfg.get('BATCH_SIZE', 16)
        
        # Calculate when this stage should end
        cumulative_target_epoch += stage_epochs

        # Define save directory for this stage
        current_save_dir = os.path.join(cfg.CHECKPOINT_DIR, f"stage_{stage_idx}_res{res}")
        last_save_dir = current_save_dir # Update tracker
        
        
      

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
        # Note: We pass 'distributed' bool or let get_loaders detect it
        train_loader, val_loader = get_loaders_for_stage(
            cfg, 
            args.dataset_name,
            resolution=res, 
            batch_size=batch_size, 
            rank=local_rank
        )

        # Skip if we resumed past this stage
        if trainer.epoch > cumulative_target_epoch:
            if is_main_process():
                console.print(f"[dim]Skipping Stage {stage_idx+1} (Res {res}) - Already completed.[/]")
                continue
                
        # --- B. Train ---
        # trainer.fit runs from current epoch -> cumulative_target_epoch
        trainer.fit(
            train_loader, 
            val_loader, 
            max_epochs=cumulative_target_epoch, 
            save_dir=current_save_dir
        )
        
        # --- C. Checkpoint Stage ---
        trainer.save_checkpoint(
            os.path.join(cfg.CHECKPOINT_DIR, f"stage_{stage_idx}_finished.pt")
        )

    # ---------------------------------------------------------
    # FINAL EVALUATION
    # ---------------------------------------------------------
    # Run the test on the very last stage's validation set and best model
    if is_main_process():
        console.print(f"[bold yellow]Looking for best model in:[/bold yellow] {last_save_dir}")
        
    if last_save_dir and val_loader:
        trainer.test(val_loader, last_save_dir)

    # --- FIX 4: Close the writer ---
    trainer.close()
    
    # --- 8. Cleanup ---
    if distributed:
        clean_ddp()
    elif is_main_process():
        console.print("[bold green]Script Finished Successfully![/]")