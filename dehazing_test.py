# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric, MetricCollection
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

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
    if args.config_file:
        try:
            cfg.merge_from_file(args.config_file)
            print(f"[+] Loaded config from {args.config_file}")
        except Exception as e:
            print(f"[-] Error merging config file {args.config_file}: {e}")

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
## Trainer
class DehazeTrainer:
    def __init__(
        self,
        cfg,
        net_G,
        net_D,
        train_loader,
        val_loader,
        device,
        vgg16_config_path="vgg16_features.json",
    ):
        self.cfg = cfg
        self.device = device
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

        self.train_loader = train_loader
        self.val_loader = val_loader

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
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.eval_metrics = MetricCollection(
            {"psnr": MeanMetric().to(self.device), "ssim": MeanMetric().to(self.device)}
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
            batch_size = x1.shape[0]

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

            #                if idx % 50 == 0:
            #                    print("[DEBUG] Batch index: ", idx)
            #                    print("[DEBUG] Loss Pixels: ", loss_pixels.item())
            #                    print("[DEBUG] Loss Flow: ", loss_flow.item())
            #                    print("[DEBUG] Loss Perceptual: ", loss_perceptual.item())
            #                    print("[DEBUG] Loss Gen: ", loss_gen.item())
            #                    print("[DEBUG] Total Loss Generative: ", loss_g.item())
            #                    print("-----------------------------------------")

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
                pred_imgs = self.ode_solver.sample(hazy_imgs)

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

        # Calcualt
        result = self.eval_metrics.compute()

        self.writer.add_scalar(
            f"Metrics/Stage_{self.stage_index}/PSNR", result["psnr"].item(), epoch
        )
        self.writer.add_scalar(
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
            # train_results = self.train_epoch(epoch)

            # --- Validation ---
            val_results = self.eval_epoch(epoch)

            print(f"Stage {stage_index} Epoch {epoch} Results:")
            # print(
            #    f"  Train: L_G={train_results['L_gen']:.4f}, L_D={train_results['L_dis']:.4f}"
            # )
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


if __name__ == "__main__":
    # 1. Parse Args
    args = create_args()

    # 2. Setup Configuration
    cfg = setup_config(args)

    # 3. Setup SEED
    set_seed(cfg.SEED)

    device = torch.device("cuda:3" if cfg.DEVICE == "cuda" else "cpu")

    print("Configuration: ")
    print(cfg)

    print(f"\n\n[+] Starting Progressive Training with {len(cfg.SCHEDULE)} stages.")
    net_G = UNet()
    net_D = Discriminator()
    # We will load the train_loader and val_loader inside the stage
    trainer = DehazeTrainer(
        cfg, net_G, net_D, train_loader=None, val_loader=None, device=device
    )

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
