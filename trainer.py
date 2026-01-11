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
from utils_ddp import is_main_process
from utils import pad_to_multiple, unpad
from model import ODESolver, path_sampler 
from data.utils import restandardize_tensor

class DehazeTrainer:
    def __init__(self, cfg, model, criterion, local_rank):
        self.cfg = cfg
        self.local_rank = local_rank 
        
        # 1. Detect Distributed Status
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # 2. Setup Device
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Setup Console (Only on Main Process)
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
                find_unused_parameters=False 
            )
            if is_main_process():
                self.console.print(Panel(f"[bold green]DDP Initialized (Rank {local_rank})[/]", title="System"))
        else:
            if is_main_process():
                self.console.print(Panel(f"[bold yellow]Single GPU Mode (No DDP)[/]", title="System"))
        
        # 5. Initialize Solver & Optimizer
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
        """Returns the underlying model regardless of DDP wrapping."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _get_progress_bar(self):
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None, style="cyan", complete_style="blue"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[bold yellow]{task.fields[info]}"),
            console=self.console,
            expand=True
        )

    def _log_visuals(self, hazy, clean, pred, step, tag="Validation_Samples/Hazy_Vs_Pred_Vs_Clean"):
        if self.writer is None:
            return
            
        # 1. Limit to top 4 samples
        #    Note: Input tensors might be on CPU from eval_epoch buffer, which is fine.
        N = min(hazy.shape[0], 4)
        
        hazy_n = hazy[:N]
        pred_n = pred[:N]
        clean_n = clean[:N]

        # 2. Interleave: [H1, P1, C1, H2, P2, C2, ...]
        # Stack dim 1 -> Shape (N, 3, C, H, W)
        stacked = torch.stack([hazy_n, pred_n, clean_n], dim=1)
        # Flatten -> Shape (N*3, C, H, W)
        interleaved = stacked.flatten(0, 1)

        print(stacked.shape)
        # 3. Create Grid
        # nrow=3 forces the layout: [Input, Prediction, Truth] per row
        grid = make_grid(
            interleaved, 
            nrow=3, 
            padding=10, 
            pad_value=1.0, 
            normalize=False
        )
        
        self.writer.add_image(tag, grid, step)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            if is_main_process():
                self.console.print(f"[bold red]!! Checkpoint not found at: {path}[/]")
            return False
        
        # Map location ensures we load to the correct local GPU
        map_location = {"cuda:0": f"cuda:{self.local_rank}"} if self.is_distributed else self.device
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        state_dict = checkpoint["model"]
        
        # Handle 'module.' prefix mismatch
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
        
        # Save raw_model state_dict for portability
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
                info_str = (
                    f"L:{loss.item():.3f} "
                    f"| F:{loss_dict['Flow']:.3f} "
                    f"P:{loss_dict['Phys']:.3f} "
                    f"V:{loss_dict['VGG']:.3f}"
                )
                progress.update(task_id, advance=1, info=info_str)

        if is_main_process():
            progress.stop()

        self.scheduler.step()
        return self.train_metrics.compute()
        
    @torch.no_grad()
    def eval_epoch(self, loader, log_tag="Validation_Samples/Hazy_Vs_Pred_Vs_Clean", step=None):
        self.model.eval()
        self.eval_metrics.reset()

        current_step = step if step is not None else self.epoch
        desc = "Final Test" if "Final" in log_tag else f"Epoch {self.epoch} [Eval]"
        progress = self._get_progress_bar() if is_main_process() else None

        # --- RANDOM SELECTION SETUP ---
        visuals_buffer = {"hazy": [], "clean": [], "pred": []}
        target_batch_indices = set()
        
        if is_main_process():
            # 1. Determine which batches to save
            total_batches = len(loader)
            num_visuals = 4 # Number of batches to capture
            
            # 2. Pick random indices (seed ensures this is deterministic if set globally)
            if total_batches > num_visuals:
                target_batch_indices = set(random.sample(range(total_batches), num_visuals))
            else:
                target_batch_indices = set(range(total_batches))

            progress.start()
            task_id = progress.add_task(desc, total=len(loader), info="Sampling...")

        for batch_idx, batch in enumerate(loader):
            clean_img, hazy_img = batch
            clean_img = clean_img.to(self.device, non_blocking=True)
            hazy_img = hazy_img.to(self.device, non_blocking=True)
            
            # Padding
            hazy_padded, pad_h, pad_w = pad_to_multiple(hazy_img, multiple=16)
            
            with torch.amp.autocast("cuda"):
                # Updated NFE to 20 for better visual quality as discussed
                pred_padded = self.ode_solver.sample(hazy_padded, 
                                                     nfe = 20 if "Final" in log_tag else 5) 
            
            pred_raw = unpad(pred_padded, pad_h, pad_w)

            # Restandardize
            pred_final = restandardize_tensor(pred_raw) 
            clean_final = restandardize_tensor(clean_img)
            hazy_final = restandardize_tensor(hazy_img)
            
            # Clamp for metrics
            pred_clean = torch.clamp(pred_final, 0.0, 1.0)
            clean_target = torch.clamp(clean_final, 0.0, 1.0) # Ensure target is also clamped [0,1]
            
            self.eval_metrics.update(pred_clean, clean_target)

            # --- CAPTURE LOGIC ---
            # Check if current batch_idx is in our pre-selected random set
            if is_main_process() and batch_idx in target_batch_indices:
                # Move to CPU immediately
                visuals_buffer["hazy"].append(hazy_final.cpu())
                visuals_buffer["clean"].append(clean_final.cpu())
                visuals_buffer["pred"].append(pred_final.cpu())
                
            if is_main_process():
                progress.update(task_id, advance=1, info="")

        if is_main_process():
            progress.stop()

            # Concatenate collected batches
            if len(visuals_buffer["hazy"]) > 0:
                hazy_cat = torch.cat(visuals_buffer["hazy"], dim=0)
                clean_cat = torch.cat(visuals_buffer["clean"], dim=0)
                pred_cat = torch.cat(visuals_buffer["pred"], dim=0)
                
                self._log_visuals(hazy_cat, clean_cat, pred_cat, current_step, tag=log_tag)
                
                if self.writer: self.writer.flush()
        
        return self.eval_metrics.compute()
        
        

    def fit(self, train_loader, val_loader, max_epochs, save_dir):
        """Standard Training Loop"""
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
            self.console.print(f"[bold]Training from Epoch {self.epoch} to {max_epochs}[/bold]")

        best_psnr = 0.0
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            current_lr = self.optimizer.param_groups[0]['lr']
            if is_main_process():
                self.writer.add_scalar("Train/LR", current_lr, epoch)
                
            # 1. Train
            train_res = self.train_epoch(train_loader)
            
            if is_main_process():
                for k, v in train_res.items(): 
                    self.writer.add_scalar(f"Train/{k}", v, epoch)

            # 2. Eval
            if epoch % self.cfg.EVAL.EVAL_INTERVAL == 0 or epoch == max_epochs:
                val_res = self.eval_epoch(val_loader, log_tag="Validation_Samples/Hazy_Vs_Pred_Vs_Clean")
                
                if is_main_process():
                    self.writer.add_scalar("Eval/PSNR", val_res['PSNR'], epoch)
                    self.writer.add_scalar("Eval/SSIM", val_res['SSIM'], epoch)
                    
                    table = Table(title=f"Epoch {epoch} Results")
                    table.add_column("Metric", style="magenta")
                    table.add_column("Value", style="green")
                    table.add_row("Loss Total", f"{train_res['Loss_Total'].item():.4f}")
                    table.add_row("PSNR", f"{val_res['PSNR'].item():.2f}")
                    table.add_row("SSIM", f"{val_res['SSIM'].item():.2f}")
                    self.console.print(table)

                    if val_res['PSNR'].item() > best_psnr:
                        best_psnr = val_res['PSNR'].item()
                        self.save_checkpoint(os.path.join(save_dir, "best.pt"), is_best=True)
            
            # 3. Save Latest
            self.save_checkpoint(os.path.join(save_dir, "latest.pt"))
            
            if self.is_distributed:
                dist.barrier()

    def test(self, loader, checkpoint_dir):
        """
        Loads the best model from checkpoint_dir and runs a final evaluation.
        """
        final_step = self.epoch
        
        if is_main_process():
            self.console.print(Panel(f"[bold cyan]Running Final Evaluation using Best Model from: {checkpoint_dir}[/]", 
                                     title="Final Test"))

        # 1. Sync
        if self.is_distributed:
            dist.barrier()

        # 2. Load Best Checkpoint
        best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        print(best_ckpt_path)
        if os.path.exists(best_ckpt_path):
            self.load_checkpoint(best_ckpt_path)
        elif is_main_process():
            self.console.print("[bold yellow]Warning: 'best.pt' not found. Using current weights.[/]")

        # 3. Run Eval with UNIQUE Tag
        final_res = self.eval_epoch(loader, log_tag="Final_Best_Model/Samples", step=final_step)

        # 4. Final Report
        if is_main_process():
            table = Table(title="FINAL TEST RESULTS (Best Model)")
            table.add_column("Metric", style="magenta", justify="center")
            table.add_column("Value", style="green", justify="center")

            psnr_val = final_res['PSNR']
            ssim_val = final_res['SSIM']

            table.add_row("Best PSNR", f"{final_res['PSNR'].item():.4f}")
            table.add_row("Best SSIM", f"{final_res['SSIM'].item():.4f}")
            
            self.console.print(table)
            self.console.print(f"[bold green]Training and Final Evaluation Complete. Best Model Saved at: {best_ckpt_path}[/]")

            # Log scalars to the final step as well
            if self.writer:
                self.writer.add_scalar("Test/PSNR", psnr_val, final_step)
                self.writer.add_scalar("Test/SSIM", ssim_val, final_step)
                
    def close(self):
        """Force write all logs to disk."""
        if self.writer:
            self.writer.flush()
            self.writer.close()