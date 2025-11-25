from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np


class RESIDE_Indoor(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.root_dir = Path(dataset_path)
        self.metadata_csv = pd.read_csv(self.root_dir / "metadata.csv")

        self.transform = transform
        self.data = []
        for idx, row in self.metadata_csv.iterrows():
            clean_path = self.root_dir / row["clear_image_path"]
            hazy_paths_str = row["hazy_image_paths"]
            hazy_image_paths = [
                path.strip()
                for path in hazy_paths_str.strip("[]").replace("'", "").split(",")
            ]
            list_hazy_paths = [
                self.root_dir / hazy_path for hazy_path in hazy_image_paths
            ]
            for hazy_path in list_hazy_paths:
                data_item = {"index": idx, "clean": clean_path, "hazy": hazy_path}

                self.data.append(data_item)

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
