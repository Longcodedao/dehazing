# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import MeanMetric, MetricCollection
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from tqdm.notebook import tqdm
from tqdm import tqdm
from data import (
    get_haze_transforms,
    restandardize_tensor,
    print_transform_summary,
    plotting_pair_images,
    partition_dataset,
)
from data import RESIDE_Indoor, Haze4k_Dataset, OHAZE_Dataset, DENSE_Haze_Dataset

from model.unet import UNet
from model.flow_matching import path_sampler, ODESolver
from model.adversarial import Discriminator

from losses import PerceptualLoss, AdversarialLoss

from yacs.config import CfgNode as CN
from config import get_cfg_defaults

import os
import argparse
from utils import (
    convert_cfg_to_dict,
    set_seed,
    get_loaders_for_stage,
    toggle_grad,
    pad_to_multiple,
    unpad,
)
from utils_ddp import is_main_process, setup_ddp, clean_ddp, reduce_tensor
import io
from contextlib import redirect_stdout
from memory_profiler import MemoryProfiler
import yaml
import gc


# %%
def create_args():
    parser = argparse.ArgumentParser(description="Dehaze Flow Matching Trainer")

    parser.add_argument(
        "--pretrain-config",
        default="configs/train_cfgs/pretrain_schedule.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    # --- Specific System Overrides ---
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Specify number of workers for loading the batch",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Specify the seed for initialization"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Path to save the log (Tensorboard)"
    )
    # Using int (0 or 1) is safer than store_true for overriding config files
    parser.add_argument(
        "--pin_memory",
        type=int,
        choices=[0, 1],
        default=None,
        help="Overwrites cfg.PIN_MEMORY (0=False, 1=True)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Save the checkpoint directory"
    )

    # --- specific Data Overrides ---
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Overwrites cfg.DATA.DATASET_ROOT",
    )

    # --- Resume ---
    parser.add_argument(
        "--resume",
        default="",
        help="path to checkpoint to resume from",
        type=str,
    )

    # --- General Overrides (yacs list) ---
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line (e.g., OPTIM.LR 1e-4)",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def setup_config(args):
    """
    Priority Order (Low to High):
    1. Default values in code (_C)
    2. YAML config file
    3. Specific argparse arguments (--num_workers, etc.)
    4. General argparse options (opts)
    """
    # 1. Get Defaults
    cfg = get_cfg_defaults()

    # 2. Merge YAML file
    if args.pretrain_config:
        try:
            cfg.merge_from_file(args.pretrain_config)
            print(f"[+] Loaded config from {args.pretrain_config}")
        except Exception as e:
            print(f"[-] Error merging config file {args.pretrain_config}: {e}")

    # 3. Merge Specific CLI arguments
    # We only update if the argument is NOT None (i.e., user actually typed it)
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.log_dir is not None:
        cfg.LOG_DIR = args.log_dir

    if args.pin_memory is not None:
        cfg.PIN_MEMORY = bool(args.pin_memory)

    if args.checkpoint_dir is not None:
        cfg.CHECKPOINT_DIR = args.checkpoint_dir

    if args.dataset_root is not None:
        # Note: dataset_root is nested under DATA in your config definition
        cfg.DATA.DATASET_ROOT = args.dataset_root

    if args.resume:
        cfg.TRAIN.RESUME_PATH = args.resume

    # 4. Merge General opts (Highest Priority)
    if args.opts:
        cfg.merge_from_list(args.opts)
        print(f"[+] Merged command line options: {args.opts}")

    # 5. Process Schedule (Convert dicts to CfgNode if necessary)
    new_schedule = []
    for item in cfg.SCHEDULE:
        if isinstance(item, dict):
            new_schedule.append(CN(item))
        else:
            new_schedule.append(item)
    cfg.SCHEDULE = new_schedule

    # 6. Freeze
    cfg.freeze()
    return cfg


# %%


# %%
## Trainer
class DehazeTrainer:
    def __init__(
        self,
        cfg,
        net_G,
        net_D,
        local_rank,
        vgg16_config_path="vgg16_features.json",
    ):
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        # Models to device
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)

        # Convert standard BN to SyncBN
        if dist.get_world_size() > 1:
            self.net_G = nn.SyncBatchNorm.convert_sync_batchnorm(self.net_G)
            self.net_D = nn.SyncBatchNorm.convert_sync_batchnorm(self.net_D)
            if self.local_rank == 0:
                print("[+] Converted models to use SyncBatchNorm")

        # Wrap with DDP
        # find_unused_parameters=True might be needed if not all layers are used in every forward pass
        self.net_G = DDP(self.net_G, device_ids=[local_rank], output_device=local_rank)
        self.net_D = DDP(self.net_D, device_ids=[local_rank], output_device=local_rank)

        # Note: Access underlying model for inference/sampling using .module
        self.ode_solver = ODESolver(self.net_G.module, nfe=10)

        # Declaring the Optimizers
        self.opt_G = optim.Adam(
            self.net_G.parameters(),
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )

        self.opt_D = optim.Adam(
            self.net_D.parameters(),
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
        )
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.opt_G, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.opt_D, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
        )

        # Placeholders
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None

        # Initialize Loss functions
        self.loss_adversarial = AdversarialLoss()
        self.loss_flow = nn.MSELoss()
        self.loss_pixels = nn.MSELoss()
        self.loss_perceptual = PerceptualLoss(vgg16_config_path).to(self.device)

        self.scaler = torch.amp.GradScaler(self.device)

        # Initialize metrics
        self.train_metrics = MetricCollection(
            {
                "L_gen": MeanMetric().to(self.device),
                "L_dis": MeanMetric().to(self.device),
                "L_flow": MeanMetric().to(self.device),
                "L_pixel": MeanMetric().to(self.device),
                "L_perceptual": MeanMetric().to(self.device),
            }
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none").to(
            self.device
        )
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.eval_metrics = MetricCollection(
            {"psnr": MeanMetric().to(self.device), "ssim": MeanMetric().to(self.device)}
        )

        self.epoch = 1

        # TensorBoard Setup
        if is_main_process():
            self.writer = SummaryWriter(cfg.LOG_DIR)
        else:
            self.writer = None

    def load_dataloader(self, dataloader, mode="train"):
        if mode == "train":
            self.train_loader = dataloader
        elif mode == "eval":
            self.val_loader = dataloader
        else:
            raise ValueError("Only support 2 modes ('train' and 'eval')")

    def save_checkpoints(self, path):
        if not is_main_process():
            return

        G_state_dict = self.net_G.state_dict()
        D_state_dict = self.net_D.state_dict()

        checkpoints = {
            "epoch": self.epoch,
            "stage_index": self.stage_index,
            "G_state_dict": G_state_dict,
            "D_state_dict": D_state_dict,
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_state_dict": self.scheduler_D.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoints, path)
        print(f"[+] Checkpoint saved to '{path}'")

    def load_checkpoints(self, path):
        """
        Safely loads a checkpoint.
        Uses .get() and checks keys to ensure partial checkpoints
        (e.g., inference-only weights) don't crash the training loop.
        """
        # Map location is crucial for DDP loading to avoid CUDA OOM or device mismatch
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.local_rank}

        if not os.path.exists(path):
            if is_main_process():
                print(f"[-] No checkpoint found at '{path}'. Starting from scratch.")
            return

        if is_main_process():
            print(f"[+] Loading checkpoint from '{path}'...")

        checkpoint = torch.load(path, map_location=map_location)

        # 1. Load Models (Weights)
        # strict=False allows loading even if some layers (like a new head) are missing
        if "G_state_dict" in checkpoint:
            self.net_G.module.load_state_dict(checkpoint["G_state_dict"], strict=False)
            print("    - Generator weights loaded.")

        if "D_state_dict" in checkpoint:
            self.net_D.module.load_state_dict(checkpoint["D_state_dict"], strict=False)
            print("    - Discriminator weights loaded.")

        # 2. Load Optimizers (Only if they exist - crucial for resuming training)
        if "opt_G_state_dict" in checkpoint:
            self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])

        if "opt_D_state_dict" in checkpoint:
            self.opt_D.load_state_dict(checkpoint["opt_D_state_dict"])

        # 3. Load Schedulers
        if "scheduler_G_state_dict" in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])

        if "scheduler_D_state_dict" in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint["scheduler_D_state_dict"])

        # 4. Load Epoch (Default to 0 if missing)
        self.epoch = checkpoint.get("epoch", 0) + 1
        self.stage_index = checkpoint.get("stage_index", 1)
        print(f"[+] Resuming from Epoch {self.epoch}")

    def train_epoch(self, epoch):
        self.net_G.train()
        self.net_D.train()
        self.train_metrics.reset()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        pbar = self.train_loader
        if is_main_process():
            pbar = tqdm(
                self.train_loader,
                desc=f"Stage {self.stage_index} | Epoch {epoch}",
                leave=True,
            )

        for idx, batch in enumerate(pbar):
            # x1: Clean image, x0: Hazy Image
            # I use this notation to match the Flow Matching definition
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)
            batch_size = x1.shape[0]

            clean_imgs = x1
            hazy_imgs = x0

            # ------------- Generator --------------------
            # Turn off the gradients of the Discriminator
            # Only updating the Generator only
            toggle_grad(self.net_D, requires_grad=False)
            self.opt_G.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type):
                # Randomize the timestamp (from 0 - 1 for constructing the path)
                t = torch.rand(batch_size, device=self.device)
                x_t, u_t = path_sampler(x0, x1, t)

                # Generator should take x_t, not x0 (Flow Matching)
                v_t = self.net_G(x_t, t)

                # Fast, single step approximation constructing the image
                pred_imgs = x0 + v_t

                # Training for the generator
                fake_output = self.net_D(pred_imgs)

                loss_pixels = self.loss_pixels(pred_imgs, clean_imgs)
                loss_flow = self.loss_flow(v_t, u_t)
                loss_perceptual = self.loss_perceptual(
                    pred_imgs,
                    clean_imgs,
                    content_weight=self.cfg.LOSS.PERCEPTUAL.CONTENT,
                    style_weight=self.cfg.LOSS.PERCEPTUAL.STYLE,
                    display=False,
                )
                loss_gen = self.loss_adversarial(D_out_fake=fake_output, mode="G")
                loss_g = (
                    self.cfg.LOSS.W_FLOW * loss_flow
                    + self.cfg.LOSS.W_PIXELS * loss_pixels
                    + self.cfg.LOSS.W_PERC * loss_perceptual
                    + self.cfg.LOSS.W_GEN * loss_gen
                )

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.opt_G)
            self.scaler.update()

            self.train_metrics["L_flow"].update(loss_flow.detach())
            self.train_metrics["L_pixel"].update(loss_pixels.detach())
            self.train_metrics["L_perceptual"].update(loss_perceptual.detach())

            self.train_metrics["L_gen"].update(loss_g.detach())

            self.opt_G.zero_grad(set_to_none=True)  # Check if clearing helps

            # 7. Cleanup Intermediates before Discriminator
            # This simulates what happens if we don't manage memory well

            # ------------ Discriminator ------------------
            # Turn the gradients of self.net_D on to update the Discriminator
            # Only update the Discriminator
            toggle_grad(self.net_D, requires_grad=True)
            self.opt_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type):
                detached_fake = pred_imgs.detach()
                #
                # Concatenate the Real and Fake Images to pass only once
                combined_input = torch.cat([clean_imgs, detached_fake], dim=0)
                combined_output = self.net_D(combined_input)

                current_batch_size = clean_imgs.shape[0]
                real_output = combined_output[:current_batch_size]
                fake_output = combined_output[current_batch_size:]
                loss_d = self.loss_adversarial(
                    D_out_real=real_output, D_out_fake=fake_output, mode="D"
                )

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()

            # Update local metrics

            self.train_metrics["L_dis"].update(loss_d.detach())

            if is_main_process() and isinstance(pbar, tqdm) and (idx % 10 == 0):
                pbar.set_postfix(
                    {"L_G": f"{loss_g.item():.4f}", "L_D": f"{loss_d.item():.4f}"}
                )

            del loss_flow, loss_pixels, loss_perceptual, loss_gen, v_t, u_t, x_t
            del loss_g, loss_d
            del x1, x0, hazy_imgs, clean_imgs, pred_imgs
            del real_output, fake_output, combined_output, combined_input, detached_fake

        self.scheduler_G.step()
        self.scheduler_D.step()

        # Save results preparing for TensorBoard Logging
        result = {}
        for key, metric in self.train_metrics.items():
            val = metric.compute()
            reduced_val = reduce_tensor(val)
            result[key] = reduced_val

        if is_main_process():
            # TensorBoard Logging for Training Losses
            self.writer.add_scalar(
                f"Loss/Stage_{self.stage_index}/G_Total", result["L_gen"].item(), epoch
            )
            self.writer.add_scalar(
                f"Loss/Stage_{self.stage_index}/D_Total", result["L_dis"].item(), epoch
            )
            self.writer.add_scalar(
                f"Loss/Stage_{self.stage_index}/Flow_Epoch",
                result["L_flow"].item(),
                epoch,
            )
            self.writer.add_scalar(
                f"Loss/Stage_{self.stage_index}/Pixel_Epoch",
                result["L_pixel"].item(),
                epoch,
            )
            self.writer.add_scalar(
                f"Loss/Stage_{self.stage_index}/Perceptual_Epoch",
                result["L_perceptual"].item(),
                epoch,
            )

        return result

    @torch.no_grad
    def eval_epoch(self, epoch):
        self.net_G.eval()
        self.net_D.eval()
        self.eval_metrics.reset()

        pbar = self.val_loader
        if is_main_process():
            pbar = tqdm(self.val_loader, "Evaluating", leave=True)

        for idx, batch in enumerate(pbar):
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)

            clean_imgs = x1
            hazy_imgs = x0

            padded_hazy, pad_h, pad_w = pad_to_multiple(hazy_imgs, multiple=32)

            with torch.autocast(device_type=self.device.type):
                pred_padded = self.ode_solver.sample(padded_hazy)

            pred_imgs = unpad(pred_padded, pad_h, pad_w)
            pred_original = restandardize_tensor(pred_imgs)
            target_original = restandardize_tensor(clean_imgs)

            self.eval_metrics["psnr"].update(
                self.psnr(pred_original.detach(), target_original.detach())
            )
            self.eval_metrics["ssim"].update(
                self.ssim(pred_original.detach(), target_original.detach())
            )

            psnr_value = self.eval_metrics["psnr"].compute()
            ssim_value = self.eval_metrics["ssim"].compute()

            if is_main_process() and isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {
                        "PSNR": f"{psnr_value:.4f}",
                        "SSIM": f"{ssim_value:.4f}",
                    }
                )
        # Compute the local Results
        local_psnr = self.eval_metrics["psnr"].compute()
        local_ssim = self.eval_metrics["ssim"].compute()

        result = {}
        # Reduce across all GPUs
        result["psnr"] = reduce_tensor(local_psnr)
        result["ssim"] = reduce_tensor(local_ssim)

        if is_main_process():
            self.writer.add_scalar(
                f"Metrics/Stage_{self.stage_index}/PSNR", result["psnr"].item(), epoch
            )
            self.writer.add_scalar(
                f"Metrics/Stage_{self.stage_index}/SSIM", result["ssim"].item(), epoch
            )
        return result

    def train_stage(
        self,
        stage_index,
        total_epochs,
        patience,
        checkpoint_dir,
        checkpoint_interval=1,
        eval_interval=5,
    ):
        """Runs the complete training cycle for a single stage (resolution)."""
        self.stage_index = stage_index
        best_metric = -float("inf")
        epochs_no_improve = 0

        # Load the latest checkpoint for this stage if it exists
        # We will check for the checkpoint structure: checkpoints/chkpoint_e*_s{stage_index}.pt
        self.epoch = 1
        best_checkpoint_path = os.path.join(checkpoint_dir, "best")
        os.makedirs(best_checkpoint_path, exist_ok=True)
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest")
        os.makedirs(latest_checkpoint_path, exist_ok=True)

        if os.path.exists(latest_checkpoint_path):
            stage_files = [
                f
                for f in os.listdir(latest_checkpoint_path)
                if f.endswith(f"_s{stage_index}.pt")
            ]

            if stage_files:
                latest_file = max(
                    stage_files, key=lambda f: int(f.split("_e")[1].split("_s")[0])
                )
                latest_path = os.path.join(latest_checkpoint_path, latest_file)

                self.load_checkpoints(latest_path)

        if is_main_process():
            print(
                f"[!] Stage {stage_index} starts at epoch {self.epoch} out of {total_epochs}."
            )

        # Important: Sync start epoch across processes just in case
        # (Though if they all read the file, they should be same)
        dist.barrier()

        if not self.train_loader or not self.val_loader:
            raise RuntimeError(
                "You need to add the train_loader and val_loader to train.\nHint: Using self.load_dataloader"
            )

        for epoch in range(self.epoch, total_epochs + 1):
            self.epoch = epoch

            # --- Training ---
            train_results = self.train_epoch(epoch)

            dist.barrier()
            # --- ADD THIS CLEANUP STEP ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if is_main_process():
                print(f"Stage {stage_index} Epoch {epoch} Results:")
                print(
                    f"  Train: L_G={train_results['L_gen']:.4f}, L_D={train_results['L_dis']:.4f}"
                )

            if epoch % eval_interval == 0:
                # --- Validation ---
                val_results = self.eval_epoch(epoch)

                if is_main_process():
                    print(
                        f"  Eval: PSNR={val_results['psnr']:.4f}, SSIM={val_results['ssim']:.4f}"
                    )

                # Use PSNR as the metric to track improvement
                current_metric = val_results["psnr"].item()

                # --- Checkpointing and Early Stopping ---
                if is_main_process():
                    if current_metric > best_metric:
                        best_metric = current_metric
                        epochs_no_improve = 0

                        # Save the BEST checkpoint
                        best_checkpoint_dir = os.path.join(
                            best_checkpoint_path, f"chkpoint_best_s{stage_index}.pt"
                        )
                        self.save_checkpoints(best_checkpoint_dir)
                    else:
                        epochs_no_improve += 1

            # Save the latest checkpoint
            if epoch % checkpoint_interval == 0:
                latest_checkpoint_dir = os.path.join(
                    latest_checkpoint_path, f"chkpoint_e{epoch}_s{stage_index}.pt"
                )
                self.save_checkpoints(latest_checkpoint_dir)

            stop_signal = torch.tensor(1 if epochs_no_improve >= patience else 0).to(
                self.device
            )
            dist.broadcast(stop_signal, src=0)

            if stop_signal.item() == 1:
                if is_main_process():
                    print(f"[!] Early stopping triggered at epoch {epoch}")
                break

        # Close TensorBoard writer after stage completion
        if is_main_process():
            self.writer.close()

    def debug_memory_usage(self, resolution=512, batch_size=8):
        """
        Runs a single step with heavy memory instrumentation to find the OOM cause.
        """
        print(
            f"\n{'=' * 20} STARTING MEMORY DEBUG {resolution}x{resolution} BS={batch_size} {'=' * 20}"
        )

        # 1. Setup Profiler
        profiler = MemoryProfiler(self.device)
        profiler.print_status("Start (Empty Cache)")

        # 2. Model Loading Impact
        # Ensure models are in training mode
        self.net_G.train()
        self.net_D.train()
        if is_main_process():
            profiler.print_status("After Models Loaded")

        # Print theoretical sizes
        profiler.inspect_model(self.net_G, "Generator")
        profiler.inspect_model(self.net_D, "Discriminator")
        profiler.inspect_model(self.loss_perceptual, "VGG Loss")

        # 3. Create a Dummy Batch (Simulation)
        # We simulate data to avoid dataloader overhead confusion
        if is_main_process():
            print("\n[STEP] Creating Dummy Batch...")
        x1 = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        x0 = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        profiler.print_status("Input Batch Loaded")

        # 4. Generator Forward (Flow Matching)
        if is_main_process():
            print("\n[STEP] Generator Forward Pass...")

        toggle_grad(self.net_D, requires_grad=False)
        self.opt_G.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type):
            t = torch.rand(batch_size, device=self.device)
            # Path Sampler
            x_t, u_t = path_sampler(x0, x1, t)

            # UNet Forward - This creates the huge Activation Graph
            v_t = self.net_G(x_t, t, profiler=profiler)
            pred_imgs = x0 + v_t

            # Discriminator Forward (on fake)
            fake_output = self.net_D(pred_imgs)

            # Loss Calc
            loss_pixels = self.loss_pixels(pred_imgs, x1)
            loss_flow = self.loss_flow(v_t, u_t)

            # Perceptual Loss (Often heavy due to VGG activations)
            loss_perceptual = self.loss_perceptual(
                pred_imgs, x1, content_weight=1.0, style_weight=100.0, display=False
            )

            loss_gen = self.loss_adversarial(D_out_fake=fake_output, mode="G")
            loss_g = loss_flow + loss_pixels + loss_perceptual + loss_gen

        profiler.print_status("After G Forward (Activations Stored)")

        # 5. Generator Backward
        if is_main_process():
            print("\n[STEP] Generator Backward Pass...")
        self.scaler.scale(loss_g).backward()
        profiler.print_status("After G Backward (Gradients calculated)")

        # 6. Generator Optimizer Step
        if is_main_process():
            print("\n[STEP] Generator Optimizer Step...")
        self.scaler.step(self.opt_G)
        self.scaler.update()
        self.opt_G.zero_grad(set_to_none=True)  # Check if clearing helps
        profiler.print_status("After G Optimizer (Adam States Created)")

        # 7. Cleanup Intermediates before Discriminator
        # This simulates what happens if we don't manage memory well
        del loss_g, loss_flow, loss_pixels, loss_perceptual, loss_gen, v_t, u_t, x_t
        # Keep pred_imgs needed for D training

        # 8. Discriminator Training
        if is_main_process():
            print("\n[STEP] Discriminator Loop...")
        toggle_grad(self.net_D, requires_grad=True)
        self.opt_D.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type):
            detached_fake = pred_imgs.detach()
            combined_input = torch.cat([x1, detached_fake], dim=0)

            # D Forward
            combined_output = self.net_D(combined_input)
            # D Loss
            # (Simplifying split for debug)
            loss_d = combined_output.mean()

        profiler.print_status("After D Forward")

        self.scaler.scale(loss_d).backward()
        profiler.print_status("After D Backward")

        self.scaler.step(self.opt_D)
        profiler.print_status("After D Optimizer")

        if is_main_process():
            print(f"\n{'=' * 20} DEBUG COMPLETE {'=' * 20}")


# %%
## Training Schedule
## Getting the cfg file

if __name__ == "__main__":
    # 1. DDP Init
    local_rank = setup_ddp()

    # 2. Parse Args & Config
    args = create_args()
    cfg = setup_config(args)
    set_seed(cfg.SEED + local_rank)

    if is_main_process():
        print("Configuration: ")
        print(cfg)
        print(f"\n[+] Starting Progressive Training with {len(cfg.SCHEDULE)} stages.")

    # 3. Models
    # Initialize on CPU or specific deviec first
    net_G = UNet(use_checkpoint=True)
    net_D = Discriminator()
    # We will load the train_loader and val_loader inside the stage
    trainer = DehazeTrainer(cfg, net_G, net_D, local_rank)

    if is_main_process():
        # 1. Convert CfgNode to standard dict/list
        cfg_dict = convert_cfg_to_dict(cfg)

        # 2. Dump to YAML string safely
        cfg_str = yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False)

        cfg_str = f"```yaml\n{cfg_str}\n```"
        trainer.writer.add_text("Configuration/Settings", cfg_str, 0)

    # Iterate through the schedule and confirm each stage is a CfgNode
    for i, stage in enumerate(cfg.SCHEDULE):
        stage_index = i + 1

        resolution = stage.RESOLUTION
        batch_size = stage.BATCH_SIZE
        epochs = stage.EPOCHS
        patience = stage.PATIENCE
        eval_interval = stage.EVAL_INTERVAL

        if is_main_process():
            print("\n==============================================")
            print(
                f"STAGE {stage_index}: Resolution={resolution}x{resolution}, Batch={batch_size} Epochs={epochs}"
            )
            print("==============================================")
            # ------------------------------------------------------
            # [NEW] Capture Transform Summary for THIS Stage
            # ------------------------------------------------------
            capture_buffer = io.StringIO()
            with redirect_stdout(capture_buffer):
                # We call this just to trigger the print statements.
                # We use the CURRENT stage's resolution.
                _, _, _ = get_loaders_for_stage(
                    cfg, resolution, batch_size, verbose=True
                )

            aug_summary = capture_buffer.getvalue()

            # Log to TensorBoard
            # We use 'stage_index' as the global_step so you can scroll through stages in TB
            trainer.writer.add_text(
                "Augmentations/Stage_Summary",
                f"```text\n{aug_summary}\n```",
                global_step=stage_index,
            )
            # ------------------------------------------------------

        train_loader, val_loader, train_sampler = get_loaders_for_stage(
            cfg, resolution, batch_size, verbose=False
        )
        trainer.load_dataloader(train_loader, mode="train")
        trainer.load_dataloader(val_loader, mode="eval")
        trainer.train_sampler = train_sampler

        # 2. Run the training for this stage
        trainer.train_stage(
            stage_index,
            epochs,
            patience,
            checkpoint_dir=cfg.CHECKPOINT_DIR,
            checkpoint_interval=cfg.CHECKPOINT_INTERVAL,
            eval_interval=eval_interval,
        )
        # capture_buffer = io.StringIO()
        # with redirect_stdout(capture_buffer):
        # We call this just to trigger the print statements.

        # trainer.debug_memory_usage(resolution=512, batch_size=32)

        #        avg_debug = capture_buffer.getvalue()
        #        if is_main_process():
        #            output_debug = "memory_debug.txt"
        #            with open(output_debug, "w") as f:
        #                f.write(avg_debug)

        dist.barrier()

    if is_main_process():
        print("\n[+] Progressive Training Complete.")
        trainer.writer.close()

    clean_ddp()

# %%
## Loading DenseHaze dataset

# transform_densehaze = get_haze_transforms(
#    dataset_name="DENSE-HAZE", resize_size=resize_size, split="val", verbose=True
# )
#
# dense_haze = DENSE_Haze_Dataset(
#    root_dir="dataset/dense-haze", transform=transform_densehaze
# )
# print("Length of Dense Haze dataset is: ", len(dense_haze))
#
## %%
# transform_ohaze = get_haze_transforms(
#    dataset_name="OHAZE", resize_size=resize_size, split="val", verbose=True
# )
# o_haze = OHAZE_Dataset(root_dir="dataset/o-haze/O-HAZY", transform=transform_densehaze)
# print("Length of O Haze dataset is: ", len(o_haze))
#
## %%
### Loader dataset
# train_loader = DataLoader(
#    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
# )
# val_loader = DataLoader(
#    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
# )
# dense_haze_loader = DataLoader(dense_haze, batch_size=16, shuffle=False, num_workers=4)
# o_haze_loader = DataLoader(o_haze, batch_size=16, shuffle=False, num_workers=4)
