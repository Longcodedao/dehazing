# %%
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import v2
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
from model import UNet
import json 


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
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


# %%
## Training process


resize_size = 256

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
## Implementing Flow Matching
def path_sampler(x0, x1, t):
    """
    Args:
        t: Timestamp uniformly sampled from [0, 1]: (B,)
        x0: Hazy image
        x1: Target image
    Return:
        x_t: Image transition at time t
        u_t: Velocity constant from x0 to x1
    """
    t = t.reshape(-1, 1, 1, 1)
    x_t = x0 * (1 - t) + x1 * t
    u_t = x1 - x0

    return x_t, u_t


class ODESolver:
    def __init__(self, model, nfe=20):
        self.model = model
        self.nfe = nfe

    def ode_func(self, t, x):
        # 1. Ensure the time 't' is a vector of size (B,)
        # The ODE solver passes 't' as a scaler (if batching is not done internally)
        # We must expand/broadcast it to match the batch size 'x'
        t = t.expand(x.size(0))

        # 2. Call the UNet (self.model)
        # The UNet predicts the velocity field (v_theta) given the time and the
        # image state
        v_theta = self.model(t, x)

        return v_theta

    @torch.no_grad()
    def sample(self, x_init):
        # 1. Define the time span for integration (from 0 to 1, in nfe steps)
        t_span = torch.linspace(0, 1, self.nfe, device=x_init.device)

        # 2. Define the ODE function for the solver to use
        # The solver requires a function (t, x) -> dx/dt
        # We can use the method we just defined:
        ode_func = self.ode_func

        # 3. Perform the ODE integration
        solution = odeint(
            ode_func, x_init, t_span, rtol=1e-5, atol=1e-5, method="euler"
        )
        # 4. The solution is a tensor of shape (NFE, B, C, H, W). We return the last state (t=1)
        return solution[-1]

batch_size = 32
t_train = torch.rand(batch_size)
x0 = torch.randn(batch_size, 64, 128, 128)
x1 = torch.randn(batch_size, 64, 128, 128)

model = UNet()
ode_solver = ODESolver(model)
mse_criterion = nn.MSELoss()

x_t, u_t = path_sampler(x0, x1, t_train)
pred_vf = model(x0, t_train)
loss_flow = mse_criterion(pred_vf, u_t)

print(f"Path sampler shape: {u_t.shape}")
print(f"Loss of the flow is: {loss_flow.item():.6f}")

# %%
### Coding Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, vgg16_config_path, vgg_backbone = "VGG16",
                 content_layers = ['relu3_3'],
                 style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 ):
        super().__init__()

        if vgg_backbone == "VGG16":
            backbone = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        elif vgg_backbone == "VGG19":
            backbone = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        else:
            raise ValueError("Only support backbone 'VGG16' and 'VGG19'.")

        with open(vgg16_config_path, 'r') as f:
            vgg_config = json.load(f) 

        self.content_layers_idx = [vgg_config[layer] for layer in content_layers]
        self.style_layers_idx = [vgg_config[layer] for layer in style_layers]

        max_index = max(self.content_layers_idx + self.style_layers_idx)
        self.backbone = backbone[:max_index + 1]

        for param in self.backbone.parameters():
            param.requires_grad = False 

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def extract_features(self, x):
        x = (x - self.mean) / self.std 
        features = {}

        for name, layer in self.backbone.named_children():
            x = layer(x)
            index = int(name)

            if index in self.content_layers_idx:
                features[f'content_{index}'] = x 

            if index in self.style_layers_idx:
                features[f'style_{index}'] = x  
            
        return features
    
    
    def gram_matrix(self, x):
        _, c, h, w = x.shape
        gram_matrix = torch.einsum('b c h w, b d h w -> b c d', x, x)
        gram_matrix = gram_matrix / (c * h * w)

        return gram_matrix
    
    def forward(self, predict, target, content_weight = 1.0, style_weight = 1e5):
        predict_feat = self.extract_features(predict)
        target_feat = self.extract_features(target)

        # Calculate the content loss (MSE)
        loss_content = 0 
        for content_key in predict_feat:
            if content_key.startswith('content'):
                loss_content += F.mse_loss(predict_feat[content_key], target_feat[content_key])        
        
        # Calculate the style loss (MSE)
        loss_style = 0
        for style_key in predict_feat:
            if style_key.startswith('style'):
                predict_gram = self.gram_matrix(predict_feat[style_key])
                target_gram = self.gram_matrix(target_feat[style_key])
                loss_style += F.mse_loss(predict_gram, target_gram)

        return content_weight * loss_content + style_weight * loss_style
        
vgg16_config_path = "vgg16_features.json"
loss = PerceptualLoss(vgg16_config_path, vgg_backbone = "VGG16")

out = torch.randn(32, 3, 128, 128)
pred = torch.randn(32, 3, 128, 128)

loss_result = loss(out, pred)
print(f"Loss is: {loss_result}")

# %%
