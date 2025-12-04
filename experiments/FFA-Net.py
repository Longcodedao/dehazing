# %%
import os
os.chdir('/kaggle/working/dehazing')
os.getcwd()

# %% 
!pip install yacs

# %%
import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Dict
from torch.utils.flop_counter import FlopCounterMode
from data.utils import print_transform_summary
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import os

from data import RESIDE_Indoor, RESIDE_SOTS_Indoor
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.optim as optim
from tqdm.notebook import tqdm

from data.utils import restandardize_tensor
from losses import PerceptualLoss
import gc
from memory_profiler import MemoryProfiler
from torch.utils.checkpoint import checkpoint


# %%
# Getting the CHannel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=1):
        super().__init__()

        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(
                dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PixelAttention(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2

        self.pa = nn.Sequential(
            nn.Conv2d(
                dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, 1, kernel_size=kernel_size, padding=padding, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# %%
## FFA Block Module
class FFA_Block(nn.Module):
    def __init__(self, dim, kernel_size, bias=True):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )

        # Pixel Attention
        hidden_dim = dim // 8
        self.pixel_attention = PixelAttention(dim, hidden_dim, kernel_size=1)
        self.channel_attention = ChannelAttention(dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x

        res = self.conv2(res)
        res = self.channel_attention(res)
        res = self.pixel_attention(res)

        res = res + x

        return res


# %%
# Group Block
class Group(nn.Module):
    def __init__(self, dim, kernel_size, blocks, bias=True, use_checkpoint = False):
        super().__init__()
        self.blocks = blocks
        modules = [FFA_Block(dim, kernel_size) for _ in range(blocks)]
        modules.append(
            nn.Conv2d(
                dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias
            )
        )
        self.group = nn.Sequential(*modules)
        
        # We add a flag to control checkpointing
        self.use_checkpoint = use_checkpoint 

    def forward(self, x):
        # We need a wrapper function for checkpoint, since it can't take module as input
        def _group_forward(x_in):
            res = self.group(x_in)
            res = res + x_in
            return res
        
        # Apply checkpoint only if flag is set
        if self.use_checkpoint and self.training:
            # The checkpoint function takes the wrapper and the input tensor(s)
            res = checkpoint(_group_forward, x)
        else:
            # Fall back to standard forward pass
            res = _group_forward(x)
      
        return res


# %%
# Create the FFA-Net Model
class FFA(nn.Module):
    def __init__(
        self, groups, blocks, channels=3, dim=64, kernel_size=3, padding=1, bias=True, use_checkpoint = False
    ):
        super().__init__()
        self.num_groups = groups

        self.preprocess = nn.Conv2d(
            channels, dim, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.groups = nn.ModuleList(
            [Group(dim, kernel_size=kernel_size, blocks=blocks, use_checkpoint = use_checkpoint) for _ in range(groups)]
        )
        self.channel_attention = ChannelAttention(
            dim * groups,
            dim // 16,
        )

        self.pixel_attention = PixelAttention(dim, dim // 8)

        self.postprocess = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.Conv2d(dim, 3, kernel_size=kernel_size, padding=padding, bias=bias),
        )

    def forward(self, x):
        x1 = self.preprocess(x)
        res = []

        temp = x1
        for idx, layer in enumerate(self.groups):
            out_group = layer(temp)
            res.append(out_group)
            temp = out_group

        # Shape [Batch, Groups * CHannels, H, W]
        res = torch.cat(res, dim=1)
        weighted_res = self.channel_attention(res)
        weighted_res = rearrange(
            weighted_res, "b (g c) h w -> b g c h w", g=self.num_groups
        )

        # This is mathematically equivalent to: w1 * res1 + w2 * res2 + w3* res3
        out = torch.sum(weighted_res, dim=1)

        # Apply Pixel Attention
        out = self.pixel_attention(out)

        x_out = self.postprocess(out)

        return x + x_out


# %%
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#INPUT_SHAPE = (1, 3, 240, 240)
#x = torch.randn(INPUT_SHAPE).to(device)
#ffa_net = FFA(groups=3, blocks=19).to(device)
#
#out = ffa_net(x)
#print(f"Shape of the output is: {out.shape}")
#
## %%
#print(f"Calculating FLOPs for input shape: {INPUT_SHAPE}")
#
#with FlopCounterMode(ffa_net) as flop_counter:
#    _ = ffa_net(x)
#
## Get the total FLOPs and format the output
#total_flops = flop_counter.get_total_flops()
#
## Convert FLOPs to Giga-FLOPs (GFLOPs) for a more readable number
#gflops = total_flops / 10**9
#

# %%
## Get the total parameters
#total_params = sum(p.numel() for p in ffa_net.parameters() if p.requires_grad)
#n_params = total_params / (10**6)
#
#print("-" * 40)
#print(f"Total FLOPs: {total_flops:,.0f}")
#print(f"Total GFLOPs: {gflops:.3f} G")
#print(f"Total parameters (in M): {n_params:.3f}M")
#print("-" * 40)
#
# %%
class MemoryProfiler_Single:
    def __init__(self, device):
        self.device = device
        self.last_allocated = 0
        self.last_reserved = 0

        # Reset peak stats at start
        torch.cuda.reset_peak_memory_stats(device)

    def _to_gb(self, bytes_val):
        return bytes_val / 1024**3

    def print_status(self, tag=""):
        # Force sync to get accurate reading
        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)

        delta_alloc = allocated - self.last_allocated

        print(f"\n[MEM] --- {tag} ---")
        print(
            f"   Active Used: {self._to_gb(allocated):.2f} GB (Delta: {self._to_gb(delta_alloc):+.2f} GB)"
        )
        print(f"   Cache/Resrv: {self._to_gb(reserved):.2f} GB")
        print(f"   Peak So Far: {self._to_gb(max_allocated):.2f} GB")

        self.last_allocated = allocated
        self.last_reserved = reserved

    def inspect_model(self, model, name="Model"):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size_mb = (param_size + buffer_size) / 1024**2

        print(f"[INFO] {name} Theoretical Size: {total_size_mb:.2f} MB")

# %%


def get_RESIDE_transform(
    resize_size,
    split="train",
    verbose=True,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
):
    dataset_name = "RESIDE"
    # --- Component Definitions ---
    common_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    if split == "train":
        # Discrete Rotation: 0, 90, 180, 270 (Legacy behavior)
        discrete_rotation = v2.RandomChoice(
            [
                v2.Identity(),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ]
        )

        geometric_sync_transformations = v2.Compose(
            [
                v2.RandomCrop(resize_size, pad_if_needed=True, padding_mode="reflect"),
                v2.RandomHorizontalFlip(p=0.5),
                discrete_rotation,
            ]
        )

        def train_transform(clean_img, hazy_img):
            clean_img, hazy_img = geometric_sync_transformations(clean_img, hazy_img)

            clean_img = common_transforms(clean_img)
            hazy_img = common_transforms(hazy_img)

            return clean_img, hazy_img 

        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tTrain Mode (Size: {resize_size}x{resize_size})",
                geometric_sync=geometric_sync_transformations,
                haze_only=v2.Identity(),
                common=common_transforms,
            )

        return train_transform

    else:

        def val_transform(clean_img, hazy_img):
            # 2. Common Transforms (Format & Normalize)
            clean_img = common_transforms(clean_img)
            hazy_img = common_transforms(hazy_img)

            return clean_img, hazy_img

        if verbose:
            print_transform_summary(
                name=f"Dataset: {dataset_name}\tEval Mode (Size: {resize_size}x{resize_size})",
                geometric_sync=v2.Identity(),
                haze_only=v2.Identity(),
                common=common_transforms,
            )

        return val_transform


# %% 

class DehazeTrainer_Normal:
    def __init__(
        self,
        cfg,
        net_G,
        local_rank=0,
        vgg16_config_path="vgg16_features.json",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ):
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.mean = mean
        self.std = std

        # Models to device
        self.net_G = net_G.to(self.device)

        # Placeholders
        self.train_loader = None
        self.val_loader = None

        # Initialize Loss functions
        self.loss_pixels = nn.L1Loss()
        self.loss_perceptual = PerceptualLoss(vgg16_config_path).to(self.device)

        # Initialize metrics
        self.train_metrics = MetricCollection(
            {"L_pixel": MeanMetric().to(self.device), 
             "L_perceptual": MeanMetric().to(self.device)
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

    def load_dataloader(self, dataloader, mode="train"):
        if mode == "train":
            self.train_loader = dataloader
        elif mode == "eval":
            self.val_loader = dataloader
        else:
            raise ValueError("Only support 2 modes ('train' and 'eval')")

    def reset_optimization(self, num_epochs):
        # Re-initialize Optimizers (Clears Momentum, resets LR to config base)
        self.opt_G = optim.Adam(
            self.net_G.parameters(),
            lr=self.cfg.OPTIM.LR,
            betas=(self.cfg.OPTIM.BETA1, self.cfg.OPTIM.BETA2),
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
        )

        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=num_epochs
        )

    def save_checkpoints(self, path):
        G_state_dict = self.net_G.state_dict()

        checkpoints = {
            "epoch": self.epoch,
            "G_state_dict": G_state_dict,
            "opt_G_state_dict": self.opt_G.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
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
        map_location = "cuda:0"
        if not os.path.exists(path):
            print(f"[-] No checkpoint found at '{path}'. Starting from scratch.")

        print(f"[+] Loading checkpoint from '{path}'...")

        checkpoint = torch.load(path, map_location=map_location)

        # 1. Load Models (Weights)
        # strict=False allows loading even if some layers (like a new head) are missing
        if "G_state_dict" in checkpoint:
            self.net_G.load_state_dict(checkpoint["G_state_dict"], strict=False)
            print("    - Generator weights loaded.")

        # 2. Load Optimizers (Only if they exist - crucial for resuming training)
        if "opt_G_state_dict" in checkpoint:
            self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])

        # 3. Load Schedulers
        if "scheduler_G_state_dict" in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])

        # 4. Load Epoch (Default to 0 if missing)
        self.epoch = checkpoint.get("epoch", 0) + 1
        print(f"[+] Resuming from Epoch {self.epoch}")

    def train_epoch(self, epoch):
        self.net_G.train()
        self.train_metrics.reset()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=True,
        )

        for idx, batch in enumerate(pbar):
            # x1: Clean image, x0: Hazy Image
            # I use this notation to match the Flow Matching definition
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)

            clean_imgs = x1
            hazy_imgs = x0

            # ------------- Generator --------------------
            # Turn off the gradients of the Discriminator
            # Only updating the Generator only
            self.opt_G.zero_grad(set_to_none=True)
            pred_imgs = self.net_G(hazy_imgs)

            loss_pixels = self.loss_pixels(pred_imgs, clean_imgs)
            loss_perceptual = self.loss_perceptual(pred_imgs, clean_imgs, display = False)

            loss_pixels.backward()
            self.opt_G.step()

            self.train_metrics["L_pixel"].update(loss_pixels.detach())
            self.train_metrics["L_perceptual"].update(loss_perceptual.detach())
            if idx % 10 == 0:
                pbar.set_postfix(
                    {
                        "L_pixel": f"{loss_pixels.item():.4f}",
                        "L_perceptual": f"{loss_perceptual.item():.4f}",
                    }
                )

        self.scheduler_G.step()

        # Save results preparing for TensorBoard Logging
        result = {}
        for key, metric in self.train_metrics.items():
            result[key] = metric.compute()

        return result

    @torch.no_grad
    def eval_epoch(self, epoch):
        self.net_G.eval()
        self.eval_metrics.reset()

        pbar = tqdm(self.val_loader, "Evaluating", leave=True)

        for idx, batch in enumerate(pbar):
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)

            clean_imgs = x1
            hazy_imgs = x0

            # padded_hazy, pad_h, pad_w = pad_to_multiple(hazy_imgs, multiple=32)

            pred_imgs = self.net_G(hazy_imgs)

            # pred_imgs = unpad(pred_padded, pad_h, pad_w)
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
        result["psnr"] = local_psnr
        result["ssim"] = local_ssim

        return result

    def train_stage(
        self,
        total_epochs,
        patience,
        checkpoint_dir,
        checkpoint_interval=1,
        eval_interval=5,
    ):
        """Runs the complete training cycle for a single stage (resolution)."""

        # Start with fresh and no stale momentum
        # If we find a checkpoint later, the optimizer and scheduler state gets overwritten
        self.reset_optimization(total_epochs)

        best_psnr_metric = -float("inf")
        best_ssim_metric = -float("inf")
        epochs_no_improve = 0

        # Load the latest checkpoint for this stage if it exists
        # We will check for the checkpoint structure: checkpoints/chkpoint_e*_s{stage_index}.pt
        self.epoch = 1
        best_checkpoint_path = os.path.join(checkpoint_dir, "best")
        os.makedirs(best_checkpoint_path, exist_ok=True)
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest")
        os.makedirs(latest_checkpoint_path, exist_ok=True)

        if os.path.exists(latest_checkpoint_path):
            list_files = os.listdir(latest_checkpoint_path)
            
            if list_files:
                latest_file = max(list_files, key=lambda f: int(f.split("_e")[1]))
                latest_path = os.path.join(latest_checkpoint_path, latest_file)

                self.load_checkpoints(latest_path)

        print(f"[!] Starts at epoch {self.epoch} out of {total_epochs}.")

        if not self.train_loader or not self.val_loader:
            raise RuntimeError(
                "You need to add the train_loader and val_loader to train.\nHint: Using self.load_dataloader"
            )

        for epoch in range(self.epoch, total_epochs + 1):
            self.epoch = epoch

            # --- Training ---
            train_results = self.train_epoch(epoch)

            # --- ADD THIS CLEANUP STEP ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Epoch {epoch} Results:")
            print(
                f"  Train: Loss Pixels={train_results['L_pixel']:.4f}, Loss Perceptual={train_results['L_perceptual']:.4f}"
            )

            if epoch % eval_interval == 0:
                # --- Validation ---
                val_results = self.eval_epoch(epoch)

                print(
                    f"  Eval: PSNR={val_results['psnr']:.4f}, SSIM={val_results['ssim']:.4f}"
                )

                # Use PSNR as the metric to track improvement
                current_psnr_metric = val_results["psnr"].item()
                current_ssim_metric = val_results["ssim"].item()

                # --- Checkpointing and Early Stopping ---
                if (
                    current_psnr_metric > best_psnr_metric
                    and current_ssim_metric > best_ssim_metric
                ):
                    best_psnr_metric = current_psnr_metric
                    best_ssim_metric = current_psnr_metric
                    epochs_no_improve = 0

                    # Save the BEST checkpoint
                    best_checkpoint_dir = os.path.join(
                        best_checkpoint_path, "chkpoint_best.pt"
                    )
                    self.save_checkpoints(best_checkpoint_dir)
                else:
                    epochs_no_improve += 1

            # Save the latest checkpoint
            if epoch % checkpoint_interval == 0:
                latest_checkpoint_dir = os.path.join(
                    latest_checkpoint_path, f"chkpoint_e{epoch}.pt"
                )
                self.save_checkpoints(latest_checkpoint_dir)

            stop_signal = torch.tensor(1 if epochs_no_improve >= patience else 0).to(
                self.device
            )
            if stop_signal.item() == 1:
                print(f"[!] Early stopping triggered at epoch {epoch}")

                break

    def debug_memory_usage(self, resolution=240, batch_size = 4):
        """
        Runs a single step with heavy memory instrumentation to find the OOM cause.
        """
        print(
            f"\n{'=' * 20} STARTING MEMORY DEBUG {resolution}x{resolution} BS={batch_size} {'=' * 20}"
        )
        self.reset_optimization(num_epochs = self.cfg.TRAIN.TOTAL_EPOCHS) 
        # 1. Setup Profiler
        profiler = MemoryProfiler_Single(self.device)
        profiler.print_status("Start (Empty Cache)")

        # 2. Model Loading Impact
        # Ensure models are in training mode
        self.net_G.train()
        profiler.print_status("After Models Loaded")

        # Print theoretical sizes
        profiler.inspect_model(self.net_G, "Generator")
        # profiler.inspect_model(self.loss_perceptual, "VGG Loss")

        # 3. Create a Dummy Batch (Simulation)
        # We simulate data to avoid dataloader overhead confusion
        print("\n[STEP] Creating Dummy Batch...")
        x1 = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        x0 = torch.randn(batch_size, 3, resolution, resolution, device=self.device)
        profiler.print_status("Input Batch Loaded")

        # 4. Generator Forward (Flow Matching)
        print("\n[STEP] Generator Forward Pass...")

        self.opt_G.zero_grad(set_to_none=True)
        
        pred_imgs = self.net_G(x0)
        loss_pixels = self.loss_pixels(pred_imgs, x0)

        profiler.print_status("After G Forward (Activations Stored)")

        # 5. Generator Backward
        print("\n[STEP] Generator Backward Pass...")
        loss_pixels.backward() 
        profiler.print_status("After G Backward (Gradients calculated)")

        print(f"\n{'=' * 20} DEBUG COMPLETE {'=' * 20}")



# %%
from yacs.config import CfgNode as CN

# Define a default configuration object
_C = CN()

# --- Optimization Hyperparameters ---
_C.OPTIM = CN()

# Learning rate for both Generator and Discriminator (if used)
_C.OPTIM.LR = 1e-4

# Adam optimizer Beta 1 parameter
_C.OPTIM.BETA1 = 0.9

# Adam optimizer Beta 2 parameter
_C.OPTIM.BETA2 = 0.999  # Standard default is 0.999, which is used in your code context

# Weight decay (L2 regularization)
_C.OPTIM.WEIGHT_DECAY = 1e-4

# --- General Training Parameters (Used elsewhere in your code) ---

# Total number of epochs to train for
_C.TRAIN = CN()
_C.TRAIN.TOTAL_EPOCHS = 500

# Number of epochs with no improvement after which training will be stopped
_C.TRAIN.PATIENCE = 50

# How often to save a checkpoint (in epochs)
_C.TRAIN.CHECKPOINT_INTERVAL = 1

# How often to run evaluation (in epochs)
_C.TRAIN.EVAL_INTERVAL = 5

# --- Data/Model Configuration (Placeholder for completeness) ---
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 8 
_C.DATA.RESOLUTION = 240  # Example
_C.DATA.MEAN = [0.64, 0.6, 0.58]
_C.DATA.STD = [0.14, 0.15, 0.152]
_C.DATA.DATASET_ROOT = "/kaggle/input"

_C.DEVICE = "cuda"
_C.NUM_WORKERS = 4
_C.SEED = 42
_C.PIN_MEMORY = False
_C.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
_C.CHECKPOINT_INTERVAL = 25


def get_cfg_defaults():
    """Get a copy of the default configuration structure."""
    return _C.clone()


# %%

cfg = get_cfg_defaults()
dataset_root = cfg.DATA.DATASET_ROOT
reside_indoor_path = "indoor-training-set-its-residestandard"
reside_sots_path = "synthetic-objective-testing-set-sots-reside"
verbose = True

train_transform_reside = get_RESIDE_transform(
    resize_size=cfg.DATA.RESOLUTION,
    split="train",
    verbose=verbose,
    mean=cfg.DATA.MEAN,
    std=cfg.DATA.STD,
)

val_transform_reside = get_RESIDE_transform(
    resize_size=cfg.DATA.RESOLUTION,
    split="val",
    verbose=verbose,
    mean=cfg.DATA.MEAN,
    std=cfg.DATA.STD,
)

train_dataset = RESIDE_Indoor(
    dataset_path=os.path.join(dataset_root, reside_indoor_path),
    transform=train_transform_reside,
)
val_dataset = RESIDE_SOTS_Indoor(
    dataset_path=os.path.join(dataset_root, reside_sots_path),
    transform=val_transform_reside,
    metadata="metadata_indoor.csv",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.DATA.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.PIN_MEMORY,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.DATA.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=cfg.PIN_MEMORY,
)

net_G = FFA(groups=3, blocks=19, use_checkpoint = True)
trainer = DehazeTrainer_Normal(cfg, net_G, mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
trainer.load_dataloader(train_loader, mode="train")
trainer.load_dataloader(val_loader, mode="eval")

# trainer.debug_memory_usage(resolution = 240, batch_size = 8)
trainer.train_stage(
    total_epochs=cfg.TRAIN.TOTAL_EPOCHS,
    patience=cfg.TRAIN.PATIENCE,
    checkpoint_dir=cfg.CHECKPOINT_DIR,
    checkpoint_interval=cfg.CHECKPOINT_INTERVAL,
    eval_interval=1,
)

# %%
