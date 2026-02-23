# experiments/ablation/base_ablation.py
import os
import sys
import argparse
import torch
import torch.distributed as dist
from yacs.config import CfgNode as CN
from torch.utils.tensorboard import SummaryWriter

# Make sure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
# Add project root to sys.path so imports work

from main import (
    create_args,
    setup_config,
    get_cfg_defaults,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_ddp,
    clean_ddp,
    get_loaders_for_stage,
)

from trainer import DehazeTrainer
from model.fmphys_mamba import FM_PhysMamba_UNET
from losses import FM_PhysicalLoss
from utils import set_seed, get_eval_loader  # if needed

from rich.console import Console
from rich.panel import Panel

console = Console() if is_main_process() else None


def build_training_config(
    pretrain_config_path: str,
    model_config: str = "small",
    dataset_name: str = "NH-HAZE",
    dataset_root: str = None,
    num_workers: int = None,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    extra_overrides: dict = None,      # ablation-specific dict
    extra_yaml_path: str = None        # extra YAML override
) -> CN:
    """
    Standalone config builder — same logic as setup_config(args), but no argparse dependency.
    Returns frozen CfgNode ready for trainer/model.
    """
    cfg = get_cfg_defaults()

    # 1. Merge schedule / pretrain YAML (same as main.py line ~78+)
    if pretrain_config_path and os.path.exists(pretrain_config_path):
        print("Exists pretrain config path")
        cfg.merge_from_file(pretrain_config_path)

    # # 2. Merge model config (small/large or path)
    # if model_config in ["small", "large"]:
    #     model_cfg_path = f"configs/model_cfgs/{model_config}.yaml"
    # else:
    #     model_cfg_path = model_config
        
    # if os.path.exists(model_cfg_path):
    #     model_cfg = CN.load_cfg(open(model_cfg_path)) 
    #     cfg.merge_from_other_cfg(model_cfg)

    # 3. Apply common overrides (same as your setup_config)
    if num_workers is not None:
        cfg.TRAIN.NUM_WORKERS = num_workers
    if seed is not None:
        cfg.TRAIN.SEED = seed
    if gradient_checkpointing:
        cfg.TRAIN.GRADIENT_CHECKPOINTING = True
    if dataset_name:
        cfg.DATA.NAME = dataset_name
    if dataset_root:
        cfg.DATA.DATASET_ROOT = dataset_root

    # 4. Apply ablation-specific dict overrides
    if extra_overrides:
        cfg.merge_from_other_cfg(CN(extra_overrides))

    # 5. Apply extra YAML if provided
    if extra_yaml_path and os.path.exists(extra_yaml_path):
        cfg.merge_from_file(extra_yaml_path)

    return cfg
    

def parse_ablation_args():
    parser = argparse.ArgumentParser(description="Ablation runner based on main.py")
    
    parser.add_argument('--ablation-name', type=str, required=True,
                        help='Name of the ablation (for logging/folder naming)')
    
    parser.add_argument('--base-config', type=str, default="configs/train_cfgs/pretrain_schedule.yaml",
                        help='Base pretrain/schedule config YAML')
    
    parser.add_argument('--model-config', type=str, default="small",
                        help='Model config ("small"/"large" or path)')
    
    parser.add_argument('--override-config', type=str, default=None,
                        help='Path to YAML file with overrides (merged last)')
    
    parser.add_argument('--override-dict', type=str, default=None,
                        help='String like \'{"LOSS.W_PHYS":0.0, "TRAIN.EPOCHS":150}\' (eval\'d as dict)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed override')
    
    parser.add_argument('--log-dir-suffix', type=str, default="",
                        help='Extra suffix for log/checkpoint dir')
    
    # Reuse some common flags from main.py
    parser.add_argument("--dataset-name", type=str, default="NHHAZE")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    
    return parser.parse_args()



def main():
    args = parse_ablation_args()

    # Set seed early
    set_seed(args.seed)

    # ───────────────────────────────────────────────────────────────
    # 1. Build base config (same as main.py)
    # ───────────────────────────────────────────────────────────────
    cfg = build_training_config(
        pretrain_config_path = args.base_config,
        model_config         = args.model_config,
        dataset_name         = args.dataset_name,
        dataset_root         = args.dataset_root,
        num_workers          = args.num_workers,
        seed                 = args.seed,
        gradient_checkpointing = args.gradient_checkpointing,
        # Ablation-specific overrides come from --override-config or --override-dict
        extra_overrides      = eval(args.override_dict) if args.override_dict else None,
        extra_yaml_path      = args.override_config
    )

    # ───────────────────────────────────────────────────────────────
    # 2. Logging / Run name
    # ───────────────────────────────────────────────────────────────
    ablation_tag = f"ablation_{args.ablation_name}"
    if args.log_dir_suffix:
        ablation_tag += f"_{args.log_dir_suffix}"

    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR or "runs", ablation_tag)
    cfg.CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR or "checkpoints", ablation_tag)
    
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    # Freeze the configuration after updating the configurations for all
    cfg.freeze()

    # Create writer early (trainer might already have one, but we can create temp one if needed)
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)  # reuse cfg.LOG_DIR
    
    # 1. Log the entire config as a nicely formatted YAML string
    config_text = cfg.dump()  # yacs CfgNode .dump() gives clean YAML
    
    writer.add_text("Config/Full", config_text, global_step=0)
    
    # 2. (Optional but very useful) Log key sections as separate scalars/text
    writer.add_text("Config/Schedule", str(cfg.SCHEDULE), 0)
    loss_weights = {
        "W_FLOW": cfg.LOSS.W_FLOW,
        "W_PERC": cfg.LOSS.W_PERC,
        "W_PHYS": cfg.LOSS.W_PHYS,
        "W_TV":   cfg.LOSS.W_TV,
        "W_ATM":  cfg.LOSS.W_ATM,
        "W_FFT":  cfg.LOSS.W_FFT,
        "W_CR":   cfg.LOSS.W_CR,
        "W_SSIM": cfg.LOSS.W_SSIM,
        "DENSITY_BOOST": cfg.LOSS.DENSITY_BOOST
    }
    
    writer.add_text("Config/Loss_Weights", str(loss_weights), 0)
    writer.add_text("Config/Optimizer", f"LR={cfg.OPTIM.LR}, Scheduler={cfg.SCHEDULER.TYPE}", 0)
    writer.add_text("Config/Train", f"Steps/epoch={cfg.TRAIN.STEPS_PER_EPOCH}, Patience={cfg.TRAIN.PATIENCE}", 0)

    # Bonus: HParams tab for easy ablation comparison
    hparams = {
        "ablation_name": args.ablation_name,
        "seed": args.seed,
        "dataset": args.dataset_name,
        **loss_weights,
        "lr": cfg.OPTIM.LR,
        "scheduler": cfg.SCHEDULER.TYPE,
        "steps_per_epoch": cfg.TRAIN.STEPS_PER_EPOCH,
    }
    writer.add_hparams(hparams, {"hparams/dummy": 0.0})  # dummy to force tab

    writer.flush()   # Ensure it's written early

    # ───────────────────────────────────────────────────────────────
    # 4. Print setup info (console)
    # ───────────────────────────────────────────────────────────────
    if is_main_process():
        console.print(Panel(
            f"[bold cyan]Ablation Run: {args.ablation_name}[/]\n"
            f"Config path: {args.base_config}\n"
            f"Overrides: {args.override_config or args.override_dict or 'None'}\n"
            f"Log dir: {cfg.LOG_DIR}\n"
            f"Checkpoint dir: {cfg.CHECKPOINT_DIR}\n"
            f"Seed: {args.seed}",
            title="Ablation Setup"
        ))
    
    # ───────────────────────────────────────────────────────────────
    # 3. Distributed setup (same as main.py)
    # ───────────────────────────────────────────────────────────────
    distributed = is_dist_avail_and_initialized()
    local_rank = 0
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_ddp(local_rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    # ───────────────────────────────────────────────────────────────
    # 4. Model & Criterion
    # ───────────────────────────────────────────────────────────────
    model = FM_PhysMamba_UNET(
        model_cfg_path = args.model_config,        
        gradient_checkpointing = cfg.TRAIN.GRADIENT_CHECKPOINTING,
        use_version = 2
    )  # assuming this is how you init
    model = model.to(device)
    
    criterion = FM_PhysicalLoss(cfg)  # pass config if needed
    criterion = criterion.to(device)

    # ───────────────────────────────────────────────────────────────
    # 5. Trainer
    # ───────────────────────────────────────────────────────────────
    trainer = DehazeTrainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        local_rank=local_rank
    )

    # ───────────────────────────────────────────────────────────────
    # 6. Progressive Training Loop (copied/adapted from main.py)
    # ───────────────────────────────────────────────────────────────
    schedule = cfg.SCHEDULE  # list of dicts {RESOLUTION, EPOCHS, BATCH_SIZE, ...}
    print(schedule)
    cumulative_target_epoch = 0
    
    for stage_idx, stage_cfg in enumerate(schedule):
        res = stage_cfg.get('RESOLUTION', 256)
        stage_epochs = stage_cfg.get('EPOCHS', 100)
        batch_size = stage_cfg.get('BATCH_SIZE', 8)
        
        cumulative_target_epoch += stage_epochs
        
        current_save_dir = os.path.join(
            cfg.CHECKPOINT_DIR, f"stage_{stage_idx}_res{res}"
        )
        
        if is_main_process():
            console.print(Panel(
                f"[bold cyan]Ablation Stage {stage_idx+1}/{len(schedule)}[/]\n"
                f"Resolution: {res}x{res}\n"
                f"Batch Size: {batch_size}\n"
                f"Epochs: {stage_epochs}\n"
                f"Target cumulative epoch: {cumulative_target_epoch}",
                title=f"{args.ablation_name} - Progressive Schedule"
            ))
        
        # Data loaders for this stage
        train_loader, val_loader = get_loaders_for_stage(
            cfg,
            args.dataset_name,
            resolution=res,
            batch_size=batch_size,
            rank=local_rank
        )
        
        # Skip completed stages (if resuming)
        if trainer.epoch >= cumulative_target_epoch:
            if is_main_process():
                console.print(f"[dim]Skipping stage {stage_idx+1} (already completed)[/]")
            continue
        
        # Train this stage
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=cumulative_target_epoch,
            save_dir=current_save_dir
        )
        
        # Save stage checkpoint
        trainer.save_checkpoint(
            os.path.join(cfg.CHECKPOINT_DIR, f"stage_{stage_idx}_finished.pt")
        )

        # ───────────────────────────────────────────────────────────────
    # 7. Final Evaluation (on last val loader + best model)
    # ───────────────────────────────────────────────────────────────
    last_save_dir = current_save_dir  # from last stage
    if is_main_process():
        console.print(f"[bold yellow]Final evaluation using best model from: {last_save_dir}[/]")
    
    if last_save_dir and val_loader:
        trainer.test(val_loader, last_save_dir)
    
    # ───────────────────────────────────────────────────────────────
    # 8. Cleanup
    # ───────────────────────────────────────────────────────────────
    trainer.close()
    
    if distributed:
        clean_ddp()
    
    if is_main_process():
        console.print("[bold green]Ablation Finished Successfully![/]")


if __name__ == "__main__":
    main()