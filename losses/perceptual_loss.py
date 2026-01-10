import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Computes Perceptual (Content) Loss using a frozen VGG16.
    
    CHANGES FROM ORIGINAL:
    1. Removed Style Loss (Gram Matrices) - unnecessary for Dehazing.
    2. Removed JSON dependency - hardcoded standard layers for stability.
    3. Added automatic input normalization.
    """
    def __init__(self, layer_indices = None):
        super().__init__()
        
        # Standard VGG16 Layers for Content Loss in Restoration:
        # relu1_2 (index 3), relu2_2 (index 8), relu3_3 (index 15), relu4_3 (index 22)
        if layer_indices is None:
            layer_indices = [3, 8, 15, 22]

        self.layer_indices = set(layer_indices)

        # Load VGG16 Backbone
        vgg = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1).features 
        vgg.eval()

        # Freeze parameters (we don't train VGG)
        for param in vgg.parameters():
            param.requires_grad = False 

        # Extract only the layers we need to save memory
        # We slice up to the max index we need
        max_idx = max(layer_indices)
        self.vgg_layers = vgg[:max_idx + 1]

        # ImageNet Normalizationo Constants 
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """
        Normalize inputs to the range and stats VGG expects.
        Assumes input x is in range [0, 1] or [-1, 1].
        """
        # If the range of input is [-1, 1] convert to [0, 1]
        if x.min() < 0:
             x = (x + 1.0) / 2.0

        # Normalize with ImageNet mean/std
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        # 1 Normalize
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        loss = 0.0
        x = pred_norm
        y = target_norm
        
        # 2. Pass through layers and accumulate loss
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            
            if i in self.layer_indices:
                # Use L1 loss for features (sharper than MSE)
                loss += F.l1_loss(x, y)
                
        return loss