import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

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
# Assuming these exist in your project structure
from utils_ddp import is_main_process
from utils import pad_to_multiple, unpad
from model import ODESolver, path_sampler 

class DehazeTrainer:
    def __init__(self, cfg, model, criterion, local_rank):
        self.cfg = cfg
        self.local_rank = local_rank 
        
        # 1. Detect Distributed Status
        # We check if torch.distributed is initialized. 
        # If running via "python main.py", this is usually False.
        # If running via "torchrun ...", this is True.
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # 2. Setup Device
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            # Fallback for single GPU debugging
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Setup Console (Only on Main Process)
        # In single GPU, local_rank is usually 0, so this works.
        self.console = Console() if is_main_process() else None
          
        # --- Model setup ---
        self.model = model.to(self.device)
        
        # 4. Conditional DDP Wrapping
        if self.is_distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(
                self.model, 
                device_ids=[local_rank],
                output_device=local_rank,
                # Set find_unused_parameters=True if your model has conditional branches
                # that might not execute every forward pass.
                find_unused_parameters=False 
            )
            if is_main_process():
                self.console.print(Panel(f"[bold green]DDP Initialized (Rank {local_rank})[/]", title="System"))
        else:
            if is_main_process():
                self.console.print(Panel(f"[bold yellow]Single GPU Mode (No DDP)[/]", title="System"))
        
        # 5. Initialize Solver
        # We pass self.raw_model (see property below) so the solver gets the actual UNet, 
        # not the DDP wrapper.
        self.criterion = criterion.to(self.device)
        self.scalar = torch.amp.GradScaler(self.device)
        self.ode_solver = ODESolver(self.raw_model)
        
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

    @property
    def raw_model(self):
        """
        Returns the underlying model regardless of whether DDP is used.
        This fixes the 'module' attribute error on single GPU.
        """
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _get_progress_bar(self):
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="blue"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold yellow]{task.fields[info]}"),
            console=self.console
        )

    def _log_visuals(self, hazy, clean, pred, step):
        if self.writer is None:
            return
        N = min(hazy.shape[0], 4)
        combined = torch.cat([hazy[:N], pred[:N], clean[:N]], dim=3)
        grid = make_grid(combined, nrow=1, padding=10, pad_value=1.0, normalize=False)
        self.writer.add_image("Validation_Samples/Hazy_Vs_Pred_Vs_Clean", grid, step)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            if is_main_process():
                self.console.print(f"[bold red]!! Checkpoint not found at: {path}[/]")
            return False
        
        map_location = {"cuda:0": f"cuda:{self.local_rank}"} if self.is_distributed else self.device
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        state_dict = checkpoint["model"]
        
        # Flexible loading: Handle 'module.' prefix mismatch
        # If checkpoint has 'module.' but current model doesn't (Single GPU): Remove it
        # If checkpoint lacks 'module.' but current model has it (DDP): Add it
        has_module_ckpt = list(state_dict.keys())[0].startswith("module.")
        is_ddp_model = hasattr(self.model, "module")
        
        if has_module_ckpt and not is_ddp_model:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        elif not has_module_ckpt and is_ddp_model:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=True)
        
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            except Exception as e:
                if is_main_process():
                     self.console.print(f"[bold yellow]Scheduler mismatch. Resetting. {e}[/]")

        self.epoch = checkpoint.get("epoch", 0) + 1
        
        if is_main_process():
            self.console.print(f"[bold green]✓ Resumed from Epoch {self.epoch}[/]")
        return True

    def save_checkpoint(self, path, is_best=False):
        if not is_main_process(): 
            return
        
        # Use self.raw_model to save clean weights without "module." prefix if desired,
        # OR keep standard DDP saving. Here we save exactly what is in self.raw_model
        # to ensure the checkpoint is portable to single GPU inference easily.
        ckpt = {
            "epoch": self.epoch,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.cfg,
        }
        
        torch.save(ckpt, path)
        if not is_best:
            self.console.print(f"[dim]Saved checkpoint: {os.path.basename(path)}[/]")

    def train_epoch(self, loader):
        self.model.train()
        self.train_metrics.reset()
        
        # FIX: Only set epoch if using DistributedSampler
        if self.is_distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.epoch)

        progress = self._get_progress_bar() if is_main_process() else None
        
        if is_main_process():
            progress.start()
            task_id = progress.add_task(f"Epoch {self.epoch} [Train]", total=len(loader), info="Init...")
        
        for batch in loader:
            clean_img, hazy_img = batch
            clean_img = clean_img.to(self.device, non_blocking=True)
            hazy_img = hazy_img.to(self.device, non_blocking=True)
            
            t = torch.rand(clean_img.shape[0], device=self.device)
            x_t, target_v = path_sampler(hazy_img, clean_img, t)

            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda"):
                preds = self.model(x_t, t)
                loss, loss_dict = self.criterion(
                    preds, target_v=target_v, x_t=x_t, timestep=t,
                    clean_img=clean_img, hazy_img=hazy_img
                )

            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

            self.train_metrics["Loss_Total"].update(loss.detach())
            self.train_metrics["Loss_Flow"].update(loss_dict["Flow"])
            self.train_metrics["Loss_Phys"].update(loss_dict["Phys"])
            self.train_metrics["Loss_VGG"].update(loss_dict["VGG"])

            if is_main_process():
                progress.update(task_id, advance=1, info=f"L: {loss.item():.4f}")

        if is_main_process():
            progress.stop()

        self.scheduler.step()
        return self.train_metrics.compute()

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        self.eval_metrics.reset()
        
        progress = self._get_progress_bar() if is_main_process() else None
        if is_main_process():
            progress.start()
            task_id = progress.add_task(f"Epoch {self.epoch} [Eval]", total=len(loader), info="Sampling...")
            target_log_batch = random.randint(0, len(loader) - 1)

        for batch_idx, batch in enumerate(loader):
            clean_img, hazy_img = batch
            clean_img = clean_img.to(self.device, non_blocking=True)
            hazy_img = hazy_img.to(self.device, non_blocking=True)
            
            hazy_padded, pad_h, pad_w = pad_to_multiple(hazy_img, multiple=16)
            
            with torch.amp.autocast("cuda"):
                # Use ode_solver which uses self.raw_model internally
                pred_padded = self.ode_solver.sample(hazy_padded, nfe=5) 
            
            pred_clean = unpad(pred_padded, pad_h, pad_w)
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)
            
            self.eval_metrics.update(pred_clean, clean_img)

            if is_main_process() and batch_idx == target_log_batch:
                self._log_visuals(hazy_img, clean_img, pred_clean, self.epoch)
                
            if is_main_process():
                progress.update(task_id, advance=1, info="")
        
        if is_main_process():
            progress.stop()
        
        return self.eval_metrics.compute()

    def fit(self, train_loader, val_loader, max_epochs, save_dir):
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
            self.console.print(f"[bold]Training from Epoch {self.epoch} to {max_epochs}[/bold]")

        best_psnr = 0.0
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            
            # 1. Train
            train_res = self.train_epoch(train_loader)
            
            if is_main_process():
                for k, v in train_res.items(): 
                    self.writer.add_scalar(f"Train/{k}", v, epoch)

            # 2. Eval
            if epoch % self.cfg.EVAL.EVAL_INTERVAL == 0 or epoch == max_epochs:
                val_res = self.eval_epoch(val_loader)
                
                if is_main_process():
                    self.writer.add_scalar("Eval/PSNR", val_res['PSNR'], epoch)
                    self.writer.add_scalar("Eval/SSIM", val_res['SSIM'], epoch)
                    
                    table = Table(title=f"Epoch {epoch} Results")
                    table.add_column("Metric", style="magenta")
                    table.add_column("Value", style="green")
                    table.add_row("Loss Total", f"{train_res['Loss_Total'].item():.4f}")
                    table.add_row("PSNR", f"{val_res['PSNR'].item():.2f}")
                    self.console.print(table)

                    if val_res['PSNR'].item() > best_psnr:
                        best_psnr = val_res['PSNR'].item()
                        self.save_checkpoint(os.path.join(save_dir, "best.pt"), is_best=True)
            
            # 3. Save Latest
            self.save_checkpoint(os.path.join(save_dir, "latest.pt"))
            
            # FIX: Only barrier if distributed. 
            # This prevents the deadlock on single GPU.
            if self.is_distributed:
                dist.barrier()