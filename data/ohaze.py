import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class OHAZE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset_dir = root_dir
        self.clear_img_paths = self.dataset_dir / "GT"
        self.hazy_img_paths = self.dataset_dir / "hazy"
        self.clear_paths = sorted(list(self.clear_img_paths.glob("*.jpg")))
        self.hazy_paths = sorted(list(self.hazy_img_paths.glob("*.jpg")))

        self.data = []

        for i in range(len(self.clear_paths)):
            data_index = {
                "index": i,
                "clean": self.clear_paths[i],
                "hazy": self.hazy_paths[i],
            }
            self.data.append(data_index)

        self.transform = transform

    def __repr__(self):
        return "O-HAZE"

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
