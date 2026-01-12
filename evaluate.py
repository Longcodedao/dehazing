import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# --- TORCHMETRICS & RICH IMPORTS ---
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# --- PROJECT IMPORTS ---
# Ensure these match your actual folder structure
from model import FM_PhysMamba_UNET, ODESolver
from data.utils import get_haze_transforms, restandardize_tensor
from data import RESIDE_SOTS_Indoor
from utils import pad_to_multiple, unpad, get_eval_loader, predict_large_image, predict_large_image_vectorized


def save_comparison(hazy_batch, pred_batch, clean_batch, idx, dataset_name, save_dir):
    """
    Robustly saves visualizations even if images have different sizes.
    Accepts either a single Batched Tensor (N, C, H, W) or a List of Tensors [(1, C, H, W), ...].
    """
    save_path = os.path.join(save_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Determine number of images
    if isinstance(hazy_batch, list):
        num_images = len(hazy_batch)
    else:
        num_images = hazy_batch.shape[0]
        
    if num_images == 0:
        return

    # Create figure: Rows = num_images
    # We increase the figure height dynamically
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    
    # Handle single-row case (matplotlib returns 1D array)
    if num_images == 1:
        axes = axes.reshape(1, -1)

    titles = ["Input (Hazy)", "Flow Matching (Ours)", "Ground Truth", "Error Map"]

    for i in range(num_images):
        # 1. Grab Tensors
        # If it's a list, we get (1, C, H, W) -> squeeze to (C, H, W)
        # If it's a batch, we get (C, H, W) directly via indexing
        if isinstance(hazy_batch, list):
            h_tensor = hazy_batch[i].squeeze(0)
            p_tensor = pred_batch[i].squeeze(0)
            c_tensor = clean_batch[i].squeeze(0)
        else:
            h_tensor = hazy_batch[i]
            p_tensor = pred_batch[i]
            c_tensor = clean_batch[i]

        def to_np(t): 
            return t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        h_img = to_np(h_tensor)
        p_img = to_np(p_tensor)
        c_img = to_np(c_tensor)
        
        # 2. Error Map
        # Note: We must resize prediction to match Clean if they differ slightly
        # (Rare, but can happen with padding issues)
        if c_img.shape != p_img.shape:
             import cv2
             p_img = cv2.resize(p_img, (c_img.shape[1], c_img.shape[0]))

        diff = np.abs(c_img - p_img)
        error_map = np.mean(diff, axis=2)

        # 3. Plotting
        items = [h_img, p_img, c_img, error_map]
        
        for col_idx, (img, title) in enumerate(zip(items, titles)):
            ax = axes[i, col_idx]
            
            if col_idx == 3: # Error Map
                im = ax.imshow(img, cmap='jet', vmin=0, vmax=0.2)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(img)
            
            if i == 0:
                ax.set_title(title, fontsize=16, fontweight='bold')
            
            ax.axis('off')

    plt.tight_layout()
    filename = f"batch_{idx:04d}_combined.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    
    # Optional: Print for confirmation (if running in a notebook/script)
    # print(f"Saved visualization batch to {full_path}")
    

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================

@torch.no_grad()
def evaluate_dataset(model, solver, dataset_name, args, console):
    loader = get_eval_loader(dataset_name, args.data_root, num_workers=args.num_workers)
    
    # Initialize Metrics
    metrics = MetricCollection({
        "PSNR": PeakSignalNoiseRatio(data_range=1.0),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0),
        "LPIPS": LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
    }).to(args.device)

    # Progress Bar
    progress = Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(), console=console
    )

    visuals_buffer = {"hazy": [], "clean": [], "pred": []}
    capture_limit = 4
    captured_count = 0

    model.eval()

    is_high_res = dataset_name.upper() in ["OHAZE", "DENSEHAZE"]
    
    with progress:
        task = progress.add_task(f"[cyan]Evaluating {dataset_name}...", total=len(loader))
        
        for batch_idx, (clean_img, hazy_img) in enumerate(loader):
            clean_img = clean_img.to(args.device)
            hazy_img = hazy_img.to(args.device)

            if is_high_res:
                pred_raw = predict_large_image_vectorized(
                    solver, 
                    hazy_img, # Pass the full resolution image
                    device=args.device,
                    progress=progress,
                    tile_size=256,   # Must match your model's training size
                    overlap_ratio=0.25, 
                    batch_size=32,    # Adjust based on your VRAM
                    nfe=args.nfe
                )
            else:
                # Inference
                hazy_padded, pad_h, pad_w = pad_to_multiple(hazy_img, multiple=16)
                with torch.amp.autocast("cuda"):
                    pred_padded = solver.sample(hazy_padded, nfe=args.nfe)
                
                pred_raw = unpad(pred_padded, pad_h, pad_w)
            
            # Post-process
            pred_final = restandardize_tensor(pred_raw).clamp(0, 1)
            clean_final = restandardize_tensor(clean_img).clamp(0, 1)
            hazy_final = restandardize_tensor(hazy_img).clamp(0, 1)

            # Update Metrics
            metrics.update(pred_final, clean_final)

            # Save Visuals (First N images only)
            if captured_count < capture_limit:
                visuals_buffer["hazy"].append(hazy_final.cpu())
                visuals_buffer["clean"].append(clean_final.cpu())
                visuals_buffer["pred"].append(pred_final.cpu())
                captured_count += 1
            
            progress.update(task, advance=1)


    # Save gathered visuals
    if len(visuals_buffer["hazy"]) > 0:
        save_comparison(
            visuals_buffer["hazy"], 
            visuals_buffer["pred"], 
            visuals_buffer["clean"], 
            0, 
            dataset_name, 
            args.output_dir
        )

    # Compute Final Metrics
    final_metrics = metrics.compute()
    return {k: v.item() for k, v in final_metrics.items()}


# ==========================================
# 3. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Dehazing Evaluation Script")
    
    # Dataset Arguments
    parser.add_argument("--datasets", nargs="+", default=["RESIDE-INDOOR"], 
                        help="List of datasets to evaluate (e.g. RESIDE-INDOOR OHAZE)")
    parser.add_argument("--data_root", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="results/eval", help="Path to save results")
    
    # Model Arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--nfe", type=int, default=10, help="Number of function evaluations (steps)")
    parser.add_argument("--model_size", type=str, default="small", help="Model size config")
    
    # System Arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    args = parser.parse_args()
    
    # Setup Console
    console = Console()
    console.rule("[bold red]FM-PhysMamba Multi-Dataset Evaluation[/bold red]")
    
    # Load Model
    console.print(f"[green]Loading Model:[/green] {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, 
                            map_location=args.device,
                            weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # Clean DDP keys
    
    model = FM_PhysMamba_UNET(args.model_size).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    solver = ODESolver(model)

    # Results Table
    results_table = Table(title="Evaluation Summary")
    results_table.add_column("Dataset", style="cyan")
    results_table.add_column("PSNR", style="green")
    results_table.add_column("SSIM", style="green")
    results_table.add_column("LPIPS", style="magenta")

    # Evaluation Loop
    for dataset in args.datasets:
        dataset = dataset.upper()
        try:
            scores = evaluate_dataset(model, solver, dataset, args, console)
            results_table.add_row(
                dataset, 
                f"{scores['PSNR']:.2f}", 
                f"{scores['SSIM']:.4f}", 
                f"{scores['LPIPS']:.4f}"
            )
        except Exception as e:
            console.print(f"[red]Error evaluating {dataset}: {e}[/red]")
            results_table.add_row(dataset, "Error", "Error", "Error")

    console.print("\n")
    console.print(results_table)
    console.print(f"\n[bold]Visualizations saved to: {args.output_dir}[/bold]")

if __name__ == "__main__":
    main()