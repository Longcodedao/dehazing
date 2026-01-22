import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import os 

class NH_Haze_Dataset(Dataset):
    def __init__(self, root_dir, transform = None, split = "train"):
        self.dataset_dir = Path(root_dir) / split
        self.clear_img_paths = self.dataset_dir / "clear"
        self.hazy_img_paths = self.dataset_dir / "hazy"
        
        self.clear_paths = sorted(list(self.clear_img_paths.glob("*.png")))
        self.hazy_paths = set(p.name for p in self.hazy_img_paths.glob("*.png"))

        self.data = []
        for i in range(len(self.clear_paths)):
            file_stem = self.clear_paths[i].stem
            
            id_image = file_stem.split("_")[0]
            hazy_filename = f"{id_image}_hazy.png"

            if hazy_filename not in self.hazy_paths:
                print(f"Warning: The id {hazy_path} does not have the hazy image")
                continue

            hazy_path = self.hazy_img_paths / hazy_filename
            
            data_index = {
                "index": i,
                "clean": self.clear_paths[i],
                "hazy": hazy_path,
            }
            self.data.append(data_index)

        self.transform = transform

    def __repr__(self):
        return "NH-HAZE"

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
            print(f"Error cannot load the file at {clean_path} and {hazy_path}")
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