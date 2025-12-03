# %%
from data.utils import get_haze_transforms, restandardize_tensor
from model.unet import UNet
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
from model.flow_matching import ODESolver


from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanMetric, MetricCollection

from utils import pad_to_multiple, unpad


# %%
# Getting the SOTS dataset
class RESIDE_SOTS_Indoor(Dataset):
    def __init__(self, dataset_path, transform=None, metadata="metadata.csv"):
        self.root_dir = Path(dataset_path)
        self.metadata_csv = pd.read_csv(self.root_dir / metadata)

        self.transform = transform
        self.data = []
        for idx, row in self.metadata_csv.iterrows():
            clean_path = self.root_dir / "indoor" / row["clear_image_path"]
            hazy_paths_str = row["hazy_image_paths"]
            hazy_image_paths = [
                path.strip()
                for path in hazy_paths_str.strip("[]").replace("'", "").split(",")
            ]
            list_hazy_paths = [
                self.root_dir / "indoor" / hazy_path for hazy_path in hazy_image_paths
            ]
            for hazy_path in list_hazy_paths:
                data_item = {"index": idx, "clean": clean_path, "hazy": hazy_path}

                self.data.append(data_item)

    def __repr__(self):
        return "RESIDE Indoor"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Missing image file at {clean_path} or {hazy_path}. Skipping")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            clean_img, hazy_img = self.transform(clean_img, hazy_img)
        else:
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )

        return clean_img, hazy_img


val_sots = get_haze_transforms(
    dataset_name="RESIDE_SOTS_Indoor", split="val", verbose=True
)
sots_indoor = RESIDE_SOTS_Indoor(
    dataset_path="dataset/reside-sots/",
    transform=val_sots,
    metadata="metadata_indoor.csv",
)

val_loader = DataLoader(
    sots_indoor, batch_size=8, shuffle=False, pin_memory=True, num_workers=4
)

# %%
device = torch.device("cuda:3")
eval_metrics = MetricCollection(
    {
        "psnr": MeanMetric().to(device),
        "ssim": MeanMetric().to(device),
    }
)
psnr_eval = PeakSignalNoiseRatio(data_range=1.0, reduction="none").to(device)
ssim_eval = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


model = UNet().to(device)
ode_solver = ODESolver(model, nfe=20)
model_path = "checkpoints/flow_matching_v1.1/best/chkpoint_best_s3.pt"
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["G_state_dict"]
new_state_dict = {}

# Iterate over all keys and remove the 'module.' prefix
for key, value in state_dict.items():
    # Only remove 'module.' prefix if it exists
    if key.startswith("module."):
        new_key = key[7:]  # Slicing from index 7 removes 'module.'
    else:
        new_key = key
    new_state_dict[new_key] = value

model = model.load_state_dict(new_state_dict, strict=False)


# %%
## Try to load the dataloader

pbar = tqdm(val_loader, "Evaluating", leave=True)
with torch.no_grad():
    for idx, batch in enumerate(pbar):
        x1, x0 = batch
        x1 = x1.to(device)
        x0 = x0.to(device)

        clean_imgs = x1
        hazy_imgs = x0

        padded_hazy, pad_h, pad_w = pad_to_multiple(hazy_imgs, multiple=32)

        with torch.autocast(device_type=device.type):
            pred_padded = ode_solver.sample(padded_hazy)

        pred_imgs = unpad(pred_padded, pad_h, pad_w)
        pred_original = restandardize_tensor(pred_imgs)
        target_original = restandardize_tensor(clean_imgs)

        eval_metrics["psnr"].update(
            psnr_eval(pred_original.detach(), target_original.detach())
        )
        eval_metrics["ssim"].update(
            ssim_eval(pred_original.detach(), target_original.detach())
        )

        psnr_value = eval_metrics["psnr"].compute()
        ssim_value = eval_metrics["ssim"].compute()

        pbar.set_postfix(
            {
                "PSNR": f"{psnr_value:.4f}",
                "SSIM": f"{ssim_value:.4f}",
            }
        )
# Compute the local Results
local_psnr = eval_metrics["psnr"].compute()
local_ssim = eval_metrics["ssim"].compute()

print(f"PSNR: {local_psnr:.2f}\tSSIM: {local_ssim:.2f}")

# %%
