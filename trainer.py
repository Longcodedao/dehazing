import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# --- Third Party Imports ---
from torchmetrics import MetricCollection, MeanMetric 
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from rich.console import Console
from rich.progress import (
    Progress, TextColumn, BarColumn, TaskProgressColumn, 
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
)
from rich.table import Table
from rich.panel import Panel

# --- Local Application Imports ---
from utils_ddp import is_main_process
from utils import pad_to_multiple, unpad
from model import ODESolver, path_sampler  # Ensure these exist in your model.py


class DehazeTrainer:
    def __init__(self, cfg, model, criterion, local_rank):
        self.cfg = cfg
        self.local_rank = local_rank 
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Setup Rich Console (Only on Main Process)
        self.console = Console() if is_main_process() else None
         
        # --- Model setup ---
        self.model = model.to(self.device)
        
        if dist.get_world_size() > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        self.model = DDP(
        self.model, 
             device_ids = [local_rank],
             output_device = local_rank, 
             find_unused_parameters = False
        )
        
        if is_main_process():
            self.console.print(Panel(f"[bold green]Model initialized on {self.device}[/]", title="System"))
            
        # Note: Access underlying model for inference/sampling using .module
        self.criterion = criterion.to(self.device)
        self.scalar = torch.amp.GradScaler(self.device)
        self.ode_solver = ODESolver(self.model.module, nfe=10)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.OPTIM.LR, 
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
        )
        
        # Metrics
        self.train_metrics = MetricCollection({
            "Loss_Total": MeanMetric(),
            "Loss_Flow": MeanMetric(),
            "Loss_Phys": MeanMetric(),
            "Loss_VGG": MeanMetric(),
        }).to(self.device)
        
        self.eval_metrics = MetricCollection({
            "PSNR": PeakSignalNoiseRatio(data_range=1.0),
            "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0),
        }).to(self.device)
        
        # Logging
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR) if is_main_process() else None
        self.epoch = 1

    
    def _get_progress_bar(self):
        """Returns a configured Rich Progress Bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="blue"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold yellow]{task.fields[info]}"), # Custom field for metrics
            console=self.console
        )


    def load_checkpoint(self, path):
        if not os.path.exists(path):
            if is_main_process():
                self.console.print(f"[bold red]!! Checkpoint not found at: {path}[/]")
            return False
        
        # Map to current device to avoid OOM 
        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        checkpoint = torch.load(path, map_location=map_location)
        
        # Handle cases where checkpoint might or might not have 'module.' prefix
        state_dict = checkpoint["model"]
        # If loading a non-DDP checkpoint into DDP, add 'module.' prefix
        if not list(state_dict.keys())[0].startswith("module."):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=True)
        
        # Load Optimizer & Scheduler
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Load Epoch
        self.epoch = checkpoint.get("epoch", 0) + 1
        
        if is_main_process():
            self.console.print(f"[bold green]✓ Resumed from Epoch {self.epoch}[/]")
        
        return True

    def save_checkpoint(self, path, is_best=False):
        if not is_main_process(): 
            return
        
        ckpt = {
            "epoch": self.epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        
        torch.save(ckpt, path)
        
        if not is_best: # Only print for regular saves to avoid spam
            self.console.print(f"[dim]Saved checkpoint: {os.path.basename(path)}[/]")

    def train_epoch(self, loader):
        self.model.train()
        self.train_metrics.reset()
        
        if hasattr(loader, "sampler") and \
            isinstance(loader.sampler, torch.utils.data.DistributedSampler):
            loader.sampler.set_epoch(self.epoch)

        # Only Main Process manages the Progress Bar
        progress = self._get_progress_bar() if is_main_process() else None
        
        # This context manager handles the display
        # We wrap the iterator so we can use 'batch' normally
        if is_main_process():
            progress.start()
            task_id = progress.add_task(f"Epoch {self.epoch} [Train]", 
                                        total=len(loader), info="Init...")
        
        for batch in loader:
            clean_img, hazy_img = batch
            clean_img = clean_img.to(self.device, non_blocking=True)
            hazy_img = hazy_img.to(self.device, non_blocking=True)
            
            # --- Flow Matching ---
            t = torch.rand(clean_img.shape[0], device=self.device)
            x_t, target_v = path_sampler(hazy_img, clean_img, t)

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda"):
                preds = self.model(x_t, t)
                loss, loss_dict = self.criterion(
                    preds, 
                    target_v = target_v,
                    x_t = x_t, 
                    timestep = t,
                    clean_img = clean_img, 
                    hazy_img = hazy_img
                )

            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

            # --- Metrics & Display ---
            self.train_metrics["Loss_Total"].update(loss.detach())
            self.train_metrics["Loss_Flow"].update(loss_dict["Flow"])
            self.train_metrics["Loss_Phys"].update(loss_dict["Phys"])
            self.train_metrics["Loss_VGG"].update(loss_dict["VGG"])

            if is_main_process():
                # Update the custom 'info' field in the progress bar
                progress.update(task_id, advance=1, 
                        info=f"L: {loss.item():.4f} | F: {loss_dict['Flow']:.4f}")

        if is_main_process():
            progress.stop()

        self.scheduler.step()
        
        # Return final computed metrics
        return self.train_metrics.compute()

        
    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        self.eval_metrics.reset()
        
        progress = self._get_progress_bar() if is_main_process() else None

        if is_main_process():
            progress.start()
            task_id = progress.add_task(f"Epoch {self.epoch} [Eval]", 
                                        total=len(loader), info="Sampling...")

        for batch in loader:
            clean_img, hazy_img = batch
            clean_img = clean_img.to(self.device, non_blocking=True)
            hazy_img = hazy_img.to(self.device, non_blocking=True)
        
            # 1. Pad to multiple of 16 (Required for UNet/Mamba architectures)
            hazy_padded, pad_h, pad_w = pad_to_multiple(hazy_img, multiple=16)
            
            with torch.amp.autocast("cuda"):
                # 2. Inference using the ODE Solver (on PADDED image)
                # nfe=5 is a good trade-off for speed during validation
                pred_padded = self.ode_solver.sample(hazy_padded, nfe=5) 
            
            # 3. Unpad the result (Critical Step!)
            # Crop the prediction back to the original size of 'clean_img'
            pred_clean = unpad(pred_padded, pad_h, pad_w)
        
            # 4. Clamp to valid image range
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)
            
            # 5. Update Metrics (Now shapes match: BxCxHxW)
            self.eval_metrics.update(pred_clean, clean_img)
            
            if is_main_process():
                progress.update(task_id, advance=1, info="")
        
        if is_main_process():
            progress.stop()
        
        return self.eval_metrics.compute()

    def fit(self, train_loader, val_loader, max_epochs,save_dir):
        """
        Args:
            max_epochs: The cumulative epoch number to stop at. 
                        (e.g., if current is 50 and max is 100, it trains for 50 epochs).
        """
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
            self.console.print(f"[bold]Training from Epoch {self.epoch} to {max_epochs}[/bold]")

        best_psnr = 0.0
        
        # Loop runs until we hit max_epochs
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            
            # 1. Train
            train_res = self.train_epoch(train_loader)
            
            # Log Train
            if is_main_process():
                for k, v in train_res.items(): 
                    self.writer.add_scalar(f"Train/{k}", v, epoch)

            # 2. Eval (Check config for interval, or force eval on last epoch)
            # You might want to pass eval_interval from the stage config if needed
            if epoch % self.cfg.EVAL.EVAL_INTERVAL == 0 or epoch == max_epochs:
                val_res = self.eval_epoch(val_loader)
                
                if is_main_process():
                    # Log Eval
                    self.writer.add_scalar("Eval/PSNR", val_res['PSNR'], epoch)
                    self.writer.add_scalar("Eval/SSIM", val_res['SSIM'], epoch)
                    
                    # Print Table
                    table = Table(title=f"Epoch {epoch} Results")
                    table.add_column("Metric", style="magenta")
                    table.add_column("Value", style="green")
                    table.add_row("Loss Total", 
                                  f"{train_res['Loss_Total'].item():.4f}")
                    table.add_row("PSNR", f"{val_res['PSNR'].item():.2f}")
                    self.console.print(table)

                    # Save Best
                    if val_res['PSNR'].item() > best_psnr:
                        best_psnr = val_res['PSNR'].item()
                        self.save_checkpoint(os.path.join(save_dir, "best.pt"), is_best=True)
            
            # 3. Save Latest
            self.save_checkpoint(os.path.join(save_dir, "latest.pt"))
            
            dist.barrier()

