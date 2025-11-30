# %%
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import v2
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchdiffeq import odeint
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
## Training process
# Loading the RESIDE Datset
train_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="train", verbose=True
)
val_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="val", verbose=True
)

reside_dataset = RESIDE_Indoor(
    dataset_path="dataset/indoor-training-set", transform=None
)
train_reside_dataset, val_reside_dataset = partition_dataset(
    reside_dataset, train_transform_reside, val_transform_reside, train_ratio=0.8
)


# Loading the Haze4k Dataset
train_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="train", verbose=True
)
val_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="val", verbose=True
)

haze_4k_train = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="train", transform=train_transform_haze4k
)

haze_4k_val = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="val", transform=val_transform_haze4k
)

train_dataset = ConcatDataset([train_reside_dataset, haze_4k_train])
val_dataset = ConcatDataset([val_reside_dataset, haze_4k_val])

# %%
print(f"Length of train dataset: {len(train_dataset)} ")
print(f"Length of valid dataset: {len(val_dataset)}")

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


## Test Loader
first_train_loader = next(iter(train_loader))
clean, hazy = first_train_loader
print(f"Clean batch has shape: {clean.shape}")
print(f"Hazy batch has shape: {hazy.shape}")


# %%
## Testing the Flow Matching
batch_size = 32
t_train = torch.rand(batch_size)
x0 = torch.randn(batch_size, 3, 128, 128)
x1 = torch.randn(batch_size, 3, 128, 128)

### Adversarial Training path
generator = UNet()
discriminator = Discriminator()

ode_solver = ODESolver(generator)
loss_flow_fn = nn.MSELoss()
loss_mse_fn = nn.MSELoss()
loss_adversarial_fn = AdversarialLoss()

vgg16_config_path = "vgg16_features.json"
loss_perceptual_fn = PerceptualLoss(vgg16_config_path, vgg_backbone="VGG16")

# Training the generator
x_t, u_t = path_sampler(x0, x1, t_train)
v_t = generator(x0, t_train)

x_reconstruct = x0 + v_t
fake_output = discriminator(x_reconstruct)

loss_mse = loss_mse_fn(x_reconstruct, x1)
loss_flow = loss_flow_fn(v_t, u_t)
loss_gen = loss_adversarial_fn(D_out_fake=fake_output, mode="G")
loss_percept = loss_perceptual_fn(predict=x_reconstruct, target=x1)

total_loss = loss_mse + loss_flow + loss_gen + loss_percept
print(f"Total Loss is: {total_loss:.4f}")
print(f"Loss MSE is: {loss_mse:.4f}")
print(f"Loss Flow Matching is: {loss_flow:.4f}")
print(f"Loss Generator is: {loss_gen:.4f}")
print(f"Loss Perceptual is: {loss_percept:.4f}")

# %%

generator = UNet()
discriminator = Discriminator()

optimizer_g = optim.Adam(
    generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2), weight_decay=WEIGHT_DECAY
)
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=20, gamma=0.5)
ode_solver = ODESolver(model=generator)


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
        self.scheduler_G = optim.lr_scheduler.StepLR(self.opt_G, step_size = cfg.SCHEDULER.STEP_SIZE, gamma = self.SCHEDULER.GAMMA)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.opt_D, step_size = cfg.SCHEDULER.STEP_SIZE, gamma = self.SCHEDULER.GAMMA)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Loss functions
        self.loss_adversarial = AdversarialLoss()
        self.loss_flow = nn.MSELoss()
        self.loss_pixels = nn.MSELoss()
        self.loss_perceptual = PerceptualLoss(vgg16_config_path)

        self.scaler = torch.amp.GradScaler(self.device)

        # Initialize metrics
        self.train_metrics = MetricCollection({
            "L_gen": MeanMetric().to(self.device),
            "L_dis": MeanMetric().to(self.device)
        })
        self.eval_metrics = MetricCollection({
            "psnr": PeakSignalNoiseRatio(data_range=1.0).to(self.device), 
            "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        })

        self.epoch = 0

    def save_checkpoints(self, path):
        G_state_dict = self.net_G.state_dict()
        D_state_dict = self.net_G.state_dict()

        checkpoints = { 
            'epoch': self.epoch,
            'G_state_dict': G_state_dict,
            'D_state_dict': D_state_dict,
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
        }
        torch.save(checkpoints, path)

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
        if 'G_state_dict' in checkpoint:
            self.net_G.load_state_dict(checkpoint['G_state_dict'], strict=False)
            print("    - Generator weights loaded.")
        else:
            print("    [!] Generator weights NOT found.")

        if 'D_state_dict' in checkpoint:
            self.net_D.load_state_dict(checkpoint['D_state_dict'], strict=False)
            print("    - Discriminator weights loaded.")

        # 2. Load Optimizers (Only if they exist - crucial for resuming training)
        if 'opt_G_state_dict' in checkpoint:
            self.opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        
        if 'opt_D_state_dict' in checkpoint:
            self.opt_D.load_state_dict(checkpoint['opt_D_state_dict'])

        # 3. Load Schedulers
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        
        if 'scheduler_D_state_dict' in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        # 4. Load Epoch (Default to 0 if missing)
        self.epoch = checkpoint.get('epoch', 0)
        print(f"[+] Resuming from Epoch {self.epoch}")


    def train_epoch(self):
        self.net_G.train()
        self.net_D.train()

        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Training with Epoch {self.epoch}")
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
                v_t = self.net_G(x0, t)

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
            with torch.autocast(device_type = self.device.type):
                real_output = self.net_D(clean_imgs)
                loss_d = self.loss_adversarial(D_out_real = real_output, D_out_fake = fake_output, mode = "D")

            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_D)
            self.scaler.update() 

            self.train_metrics['L_gen'].update(loss_g.detach())
            self.train_metrics['L_dis'].update(loss_d.detach())

            if idx % 10 == 0:
                # Compute returns the average over all batches seen so far in this epoch
                avg_g = self.train_metrics['L_gen'].compute()
                avg_d = self.train_metrics['L_dis'].compute()
                
                pbar.set_postfix({
                    "Loss G": f"{avg_g:.4f}",
                    "Loss D": f"{avg_d:.4f}",
                    "Flow": f"{loss_flow.item():.4f}" # Instantaneous value
                })

        result = self.train_metrics.compute() 
        self.scheduler_G.step()
        self.scheduler_D.step()

        return result

    @torch.no_grad 
    def eval_epoch(self):
        self.net_G.eval()
        self.net_D.eval()

        self.eval_metrics.reset()
        pbar = tqdm(self.val_loader, 'Evaluating')
        for idx, batch in enumerate(pbar):
            x1, x0 = batch 
            x1 = x1.to(self.device)
            x0 = x0.to(self.device)

            clean_imgs = x1 
            hazy_imgs = x0 

           
            with torch.autocast(device_type = self.device.type):
                pred_imgs = self.ode_solver(hazy_imgs)

            pred_original = restandardize_tensor(pred_imgs)
            target_original = restandardize_tensor(clean_imgs)

            self.eval_metrics['psnr'].compute(pred_original.detach(), target_original.detach())
            self.eval_metrics['ssim'].compute(pred_original.detach(), target_original.detach())

        # Calcualt
        result = self.eval_metrics.compute()
        return result
    
# %%
## Training Schedule
cfg = get_cfg_defaults()

# 2. Merge the specific schedule file
# This replaces the empty [] list in config.py with the list from your YAML
schedule_path = "configs/train_cfgs/pretrain_schedule.yaml"
cfg.merge_from_file(schedule_path)

# 3. Freeze the config to prevent accidental changes
cfg.freeze()

# --- Verification ---
print("Loaded Schedule:")
for stage in cfg.SCHEDULE:
    print(f"- Res: {stage.RESOLUTION} | Epochs: {stage.EPOCHS} | Batch: {stage.BATCH_SIZE}")



# %%
train_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="train", verbose=True
)
val_transform_reside = get_haze_transforms(
    dataset_name="RESIDE", resize_size=resize_size, split="val", verbose=True
)

reside_dataset = RESIDE_Indoor(
    dataset_path="dataset/indoor-training-set", transform=None
)
train_reside_dataset, val_reside_dataset = partition_dataset(
    reside_dataset, train_transform_reside, val_transform_reside, train_ratio=0.8
)


# Loading the Haze4k Dataset
train_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="train", verbose=True
)
val_transform_haze4k = get_haze_transforms(
    dataset_name="HAZE4K", resize_size=resize_size, split="val", verbose=True
)

haze_4k_train = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="train", transform=train_transform_haze4k
)

haze_4k_val = Haze4k_Dataset(
    root_dir="dataset/haze4k", split="val", transform=val_transform_haze4k
)

train_dataset = ConcatDataset([train_reside_dataset, haze_4k_train])
val_dataset = ConcatDataset([val_reside_dataset, haze_4k_val])

# %%
print(f"Length of train dataset: {len(train_dataset)} ")
print(f"Length of valid dataset: {len(val_dataset)}")

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