import torch
import torch.nn as nn 

class CharbonnierLoss(nn.Module):
    """
    Robust L1 loss (softer near zero).
    Standard for image restoration tasks.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diffs * diff + self.eps**2)
        return torch.mean(loss)