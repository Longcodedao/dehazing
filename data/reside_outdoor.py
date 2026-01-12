import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np


class RESIDE_Outdoor(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Args:
            dataset_path (str or Path): Path to the SOTS folder (containing 'outdoor' folder).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(dataset_path)
        self.transform = transform
        self.data = []

        # Standard SOTS Outdoor Structure:
        # root/outdoor/clear/ ... (files like 0001.png)
        # root/outdoor/hazy/  ... (files like 0001_0.04_0.8.jpg)
        
        # Adjust these subpaths if your folder structure is slightly different
        self.clear_dir = self.root_dir / "clear"
        self.hazy_dir = self.root_dir  / "hazy"

        if not self.hazy_dir.exists():
            raise FileNotFoundError(f"Could not find hazy directory at {self.hazy_dir}")

        # 1. Scan the Hazy Directory
        hazy_images = sorted(list(self.hazy_dir.glob("*.jpg")) + list(self.hazy_dir.glob("*.png")))

        # 2. Match with Clear Images based on ID
        for hazy_path in hazy_images:
            # Filename format: ID_Beta_A.jpg (e.g., 0001_0.04_0.8.jpg)
            # We want the ID (0001) to find the clear image.
            
            file_name = hazy_path.name
            img_id = file_name.split("_")[0] # Extracts "0001"
            
            # SOTS clear images are usually .png, but we check both just in case
            clean_path_png = self.clear_dir / f"{img_id}.png"
            clean_path_jpg = self.clear_dir / f"{img_id}.jpg"

            if clean_path_png.exists():
                clean_path = clean_path_png
            elif clean_path_jpg.exists():
                clean_path = clean_path_jpg
            else:
                # Fallback: Sometimes clear images have the same name as hazy in different folders
                # or just the ID. If we can't find it, we skip or warn.
                print(f"Warning: Clean image for ID {img_id} not found. Skipping {file_name}")
                continue

            self.data.append({
                "clean": clean_path,
                "hazy": hazy_path,
                "id": img_id
            })

    def __repr__(self):
        return f"RESIDE Outdoor ({len(self.data)} pairs)"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {hazy_path}: {e}. Skipping to next.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            clean_img, hazy_img = self.transform(clean_img, hazy_img)
        else:
            # Default transform if none provided
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )

        return clean_img, hazy_img
        

class RESIDE_SOTS_Outdoor(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Args:
            dataset_path (str or Path): Path to the SOTS folder (containing 'outdoor' folder).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(dataset_path)
        self.transform = transform
        self.data = []

        # Standard SOTS Outdoor Structure:
        # root/outdoor/clear/ ... (files like 0001.png)
        # root/outdoor/hazy/  ... (files like 0001_0.04_0.8.jpg)
        
        # Adjust these subpaths if your folder structure is slightly different
        self.clear_dir = self.root_dir / "outdoor" / "clear"
        self.hazy_dir = self.root_dir / "outdoor" / "hazy"

        if not self.hazy_dir.exists():
            raise FileNotFoundError(f"Could not find hazy directory at {self.hazy_dir}")

        # 1. Scan the Hazy Directory
        hazy_images = sorted(list(self.hazy_dir.glob("*.jpg")) + list(self.hazy_dir.glob("*.png")))

        # 2. Match with Clear Images based on ID
        for hazy_path in hazy_images:
            # Filename format: ID_Beta_A.jpg (e.g., 0001_0.04_0.8.jpg)
            # We want the ID (0001) to find the clear image.
            
            file_name = hazy_path.name
            img_id = file_name.split("_")[0] # Extracts "0001"
            
            # SOTS clear images are usually .png, but we check both just in case
            clean_path_png = self.clear_dir / f"{img_id}.png"
            clean_path_jpg = self.clear_dir / f"{img_id}.jpg"

            if clean_path_png.exists():
                clean_path = clean_path_png
            elif clean_path_jpg.exists():
                clean_path = clean_path_jpg
            else:
                # Fallback: Sometimes clear images have the same name as hazy in different folders
                # or just the ID. If we can't find it, we skip or warn.
                print(f"Warning: Clean image for ID {img_id} not found. Skipping {file_name}")
                continue

            self.data.append({
                "clean": clean_path,
                "hazy": hazy_path,
                "id": img_id
            })

    def __repr__(self):
        return f"RESIDE SOTS-Outdoor ({len(self.data)} pairs)"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        clean_path = data_item["clean"]
        hazy_path = data_item["hazy"]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            hazy_img = Image.open(hazy_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {hazy_path}: {e}. Skipping to next.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            clean_img, hazy_img = self.transform(clean_img, hazy_img)
        else:
            # Default transform if none provided
            clean_img = (
                torch.as_tensor(np.array(clean_img)).permute(2, 0, 1).float() / 255.0
            )
            hazy_img = (
                torch.as_tensor(np.array(hazy_img)).permute(2, 0, 1).float() / 255.0
            )

        return clean_img, hazy_img



