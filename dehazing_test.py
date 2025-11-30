# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric, MetricCollection
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
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

# %%
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
B1, B2 = 0.5, 0.999
W_FLOW, W_MSE, W_PERC, W_ADV = 1.0, 1.0, 0.1, 0.01
TIME_LIMIT_HOURS = 11.5
BATCH_SIZE = 256


# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
resize_size = 256


# %%
## Loading DenseHaze dataset

transform_densehaze = get_haze_transforms(
    dataset_name="DENSE-HAZE", resize_size=resize_size, split="val", verbose=True
)

dense_haze = DENSE_Haze_Dataset(
    root_dir="dataset/dense-haze", transform=transform_densehaze
)
print("Length of Dense Haze dataset is: ", len(dense_haze))

# %%
transform_ohaze = get_haze_transforms(
    dataset_name="OHAZE", resize_size=resize_size, split="val", verbose=True
)
o_haze = OHAZE_Dataset(root_dir="dataset/o-haze/O-HAZY", transform=transform_densehaze)
print("Length of O Haze dataset is: ", len(o_haze))

# %%
## Loader dataset

dense_haze_loader = DataLoader(dense_haze, batch_size=16, shuffle=False, num_workers=4)
o_haze_loader = DataLoader(o_haze, batch_size=16, shuffle=False, num_workers=4)


class DehazeTrainer:
    def __init__(
        self,
        cfg,
        net_G,
        net_D,
        train_loader,
        val_loader,
        vgg16_config_path="vgg16_features.json",
    ):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)
        self.ode_solver = ODESolver(self.net_G)

        # Declaring the Optimizers
        self.opt_G = optim.Adam(
            self.net_G.parameters(),
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        )

        self.opt_D = optim.Adam(
            self.net_G.parameters(),
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
        )
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.opt_G, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.opt_D, step_size=cfg.SCHEDULER.STEP_SIZE, gamma=cfg.SCHEDULER.GAMMA
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Loss functions
        self.loss_adversarial = AdversarialLoss()
        self.loss_flow = nn.MSELoss()
        self.loss_pixels = nn.MSELoss()
        self.loss_perceptual = PerceptualLoss(vgg16_config_path)

        self.scaler = torch.amp.GradScaler(self.device)

        # Initialize metrics
        self.train_metrics = MetricCollection(
            {
                "L_gen": MeanMetric().to(self.device),
                "L_dis": MeanMetric().to(self.device),
                "L_flow": MeanMetric().to(self.device),
                "L_pixel": MeanMetric().to(self.device),
                "L_perceptual": MeanMetric.to(self.device),
            }
        )
        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device),
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(
                    self.device
                ),
            }
        )

        self.epoch = 1

        # TensorBoard Setup
        self.writer = SummaryWriter(cfg.LOG_DIR)

    def save_checkpoints(self, path):
        G_state_dict = self.net_G.state_dict()
        D_state_dict = self.net_G.state_dict()

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
        if not os.path.exists(path):
            print(f"[-] No checkpoint found at '{path}'. Starting from scratch.")
            return

        print(f"[+] Loading checkpoint from '{path}'...")

        # Load to CPU first to prevent GPU OOM, then map to device
        checkpoint = torch.load(path, map_location=self.device)

        # 1. Load Models (Weights)
        # strict=False allows loading even if some layers (like a new head) are missing
        if "G_state_dict" in checkpoint:
            self.net_G.load_state_dict(checkpoint["G_state_dict"], strict=False)
            print("    - Generator weights loaded.")
        else:
            print("    [!] Generator weights NOT found.")

        if "D_state_dict" in checkpoint:
            self.net_D.load_state_dict(checkpoint["D_state_dict"], strict=False)
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
        self.epoch = checkpoint.get("epoch", 1)
        self.stage_index = checkpoint.get("stage_index", 1)
        print(f"[+] Resuming from Epoch {self.epoch}")

    def train_epoch(self, epoch):
        self.net_G.train()
        self.net_D.train()

        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Stage {self.stage_index} | Epoch {epoch}")
        for idx, batch in enumerate(pbar):
            # x1: Clean image, x0: Hazy Image
            # I use this notation to match the Flow Matching definition
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)
            batch_size = x1.shape

            clean_imgs = x1
            hazy_imgs = x0

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
                    style_wieght=self.cfg.LOSS.PERCEPTUAL.STYLE,
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

            self.opt_D.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type):
                real_output = self.net_D(clean_imgs)
                # Need to run the D(fake) again
                fake_output = self.net_D(pred_imgs.detach())

                loss_d = self.loss_adversarial(
                    D_out_real=real_output, D_out_fake=fake_output, mode="D"
                )

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update()

            self.train_metrics["L_gen"].update(loss_g.detach())
            self.train_metrics["L_dis"].update(loss_d.detach())
            self.train_metrics["L_flow"].update(loss_flow.detach())
            self.train_metrics["L_pixel"].update(loss_pixels.detach())
            self.train_metrics["L_perceptual"].update(loss_pixels.detach())

            if idx % 10 == 0:
                # Compute returns the average over all batches seen so far in this epoch
                avg_g = self.train_metrics["L_gen"].compute()
                avg_d = self.train_metrics["L_dis"].compute()
                avg_flow = self.train_metrics["L_flow"].compute()

                pbar.set_postfix(
                    {
                        "Loss G": f"{avg_g:.4f}",
                        "Loss D": f"{avg_d:.4f}",
                        "Flow": f"{avg_flow:.4f}",  # Instantaneous value
                    }
                )
        self.scheduler_G.step()
        self.scheduler_D.step()

        # Save results preparing for TensorBoard Logging
        result = self.train_metrics.compute()

        # TensorBoard Logging for Training Losses
        self.writer.add_scalar(
            f"Loss/Stage_{self.stage_index}/G_Total", loss_g.item(), epoch
        )
        self.writer.add_scalar(
            f"Loss/Stage_{self.stage_index}/D_Total", loss_d.item(), epoch
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
        pbar = tqdm(self.val_loader, "Evaluating")
        for idx, batch in enumerate(pbar):
            x1, x0 = batch
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)

            clean_imgs = x1
            hazy_imgs = x0

            with torch.autocast(device_type=self.device.type):
                pred_imgs = self.ode_solver(hazy_imgs)

            pred_original = restandardize_tensor(pred_imgs)
            target_original = restandardize_tensor(clean_imgs)

            self.eval_metrics["psnr"].compute(
                pred_original.detach(), target_original.detach()
            )
            self.eval_metrics["ssim"].compute(
                pred_original.detach(), target_original.detach()
            )

        # Calcualt
        result = self.eval_metrics.compute()

        self.writer.add_scaler(
            f"Metrics/Stage_{self.stage_index}/PSNR", result["psnr"].item(), epoch
        )
        self.writer.add_scaler(
            f"Metrics/Stage_{self.stage_index}/SSIM", result["ssim"].item(), epoch
        )
        return result

    def train_stage(
        self, stage_index, total_epochs, patience, checkpoint_dir, checkpoint_interval=1
    ):
        """Runs the complete training cycle for a single stage (resolution)."""
        self.stage_index = stage_index
        best_metric = -float("inf")
        epochs_no_improve = 0.0

        # Load the latest checkpoint for this stage if it exists
        # We will check for the checkpoint structure: checkpoints/chkpoint_e*_s{stage_index}.pt
        self.epoch = 1
        if os.path.exists(checkpoint_dir):
            stage_files = [
                f
                for f in os.listdir(checkpoint_dir)
                if f.endswith(f"_s{stage_index}.pt")
            ]
            if stage_files:
                latest_file = max(
                    stage_files, key=lambda f: int(f.split("_e")[1].split("_s")[0])
                )
                latest_path = os.path.join(checkpoint_dir, latest_file)

                self.load_checkpoints(self, latest_path)

        print(
            f"[!] Stage {stage_index} starts at epoch {self.epoch} out of {total_epochs}."
        )

        for epoch in range(self.epoch, total_epochs + 1):
            # --- Training ---
            train_results = self.train_epoch(epoch)

            # --- Validation ---
            val_results = self.eval_epoch(epoch)

            print(f"Stage {stage_index} Epoch {epoch} Results:")
            print(
                f"  Train: L_G={train_results['L_gen']:.4f}, L_D={train_results['L_dis']:.4f}"
            )
            print(
                f"  Eval: PSNR={val_results['psnr']:.4f}, SSIM={val_results['ssim']:.4f}"
            )

            # Use PSNR as the metric to track improvement
            current_metric = val_results["psnr"].item()

            # --- Checkpointing and Early Stopping ---
            if current_metric > best_metric:
                best_metric = current_metric
                epochs_no_improve = 0

                # Save the BEST checkpoint
                best_checkpoint_dir = os.path.join(
                    checkpoint_dir, f"chkpoint_best_s{stage_index}.pt"
                )
                self.save_checkpoints(best_checkpoint_dir)
            else:
                epochs_no_improve += 1

            # Save the latest checkpoint
            if epoch % checkpoint_interval == 0:
                latest_checkpoint_dir = os.path.join(
                    checkpoint_dir, f"chkpoint_e{epoch}_s{stage_index}.pt"
                )
                self.save_checkpoints(latest_checkpoint_dir)

            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}")
                break

        # Close TensorBoard writer after stage completion
        self.writer.close()


# %%
## Training Schedule
## Getting the cfg file
def load_pretrain_config(cfg, yaml_path):
    # 2. Merge the specific schedule file
    # This replaces the empty [] list in config.py with the list from your YAML
    try:
        cfg.merge_from_file(yaml_path)
    except Exception as e:
        print(f"Error merging config file {yaml_path}: {e}")

    new_schedule = []
    for item in cfg.SCHEDULE:
        if isinstance(item, dict):
            new_schedule.append(CN(item))
        else:
            new_schedule.append(item)
    cfg.SCHEDULE = new_schedule

    cfg.freeze()
    return cfg


def get_loaders_for_stage(cfg, resolution, batch_size):
    data_cfg = cfg.DATA

    train_transform_reside = get_haze_transforms(
        dataset_name="RESIDE",
        resize_size=resolution,
        split="train",
        verbose=True,
    )

    val_transform_reside = get_haze_transforms(
        dataset_name="RESIDE",
        resize_size=resolution,
        split="val",
        verbose=True,
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
        dataset_name="HAZE4K", resize_size=resize_size, split="train", verbose=True
    )
    val_transform_haze4k = get_haze_transforms(
        dataset_name="HAZE4K", resize_size=resize_size, split="val", verbose=True
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=stage.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader


cfg = get_cfg_defaults()
yaml_path = "configs/train_cfgs/pretrain_schedule.yaml"
cfg = load_pretrain_config(cfg, yaml_path)

print("Configuration: ")
print(cfg)

print("[+] Starting Progressive Training with {len(cfg.SCHEDULE)} stages.")
net_G = UNet()
net_D = Discriminator()
# We will load the train_loader and val_loader inside the stage
trainer = DehazeTrainer(cfg, net_G, net_D, train_loader=None, val_loader=None)

# Iterate through the schedule and confirm each stage is a CfgNode
for i, stage in enumerate(cfg.SCHEDULE):
    stage_index = i + 1

    resolution = stage.RESOLUTION
    batch_size = stage.BATCH_SIZE
    epochs = stage.EPOCHS
    patience = stage.PATIENCE

    print("\n==============================================")
    print(
        f"STAGE {stage_index}: Resolution={resolution}x{resolution}, Batch={batch_size}"
    )
    print("==============================================")

    train_loader, val_loader = get_loaders_for_stage(cfg, resolution, batch_size)
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    # 2. Run the training for this stage
    trainer.train_stage(
        stage_index,
        epochs,
        patience,
        checkpoint_dir=cfg.CHECKPOINT_DIR,
        checkpoint_interval=cfg.CHECKPOINT_INTERVAL,
    )

print("\n[+] Progressive Training Complete.")
trainer.writer.close()

# %%
## Loading DenseHaze dataset

transform_densehaze = get_haze_transforms(
    dataset_name="DENSE-HAZE", resize_size=resize_size, split="val", verbose=True
)

dense_haze = DENSE_Haze_Dataset(
    root_dir="dataset/dense-haze", transform=transform_densehaze
)
print("Length of Dense Haze dataset is: ", len(dense_haze))

# %%
transform_ohaze = get_haze_transforms(
    dataset_name="OHAZE", resize_size=resize_size, split="val", verbose=True
)
o_haze = OHAZE_Dataset(root_dir="dataset/o-haze/O-HAZY", transform=transform_densehaze)
print("Length of O Haze dataset is: ", len(o_haze))

# %%
## Loader dataset
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
dense_haze_loader = DataLoader(dense_haze, batch_size=16, shuffle=False, num_workers=4)
o_haze_loader = DataLoader(o_haze, batch_size=16, shuffle=False, num_workers=4)
