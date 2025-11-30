import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_labels(self, tensor, is_real):
        b = tensor.size(0)

        label_val = 1.0 if is_real else 0.0
        labels = torch.full((b, 1), label_val, dtype=torch.float, device=tensor.device)
        return labels

    def calculate_D_loss(self, D_out_real, D_out_fake):
        """
        Calculates the Discriminator's total loss (L_D)
        D aims to minimize this loss to correctly detect which one is the
        real output, which one is generated
        """
        # Loss on Reall Samples (Target = 1.0)
        labels_real = self._get_labels(D_out_real, is_real=True)
        loss_real = self.criterion(D_out_real, labels_real)
        labels_fake = self._get_labels(D_out_fake, is_real=False)
        loss_fake = self.criterion(D_out_fake, labels_fake)

        return loss_real + loss_fake

    def calculate_G_loss(self, D_out_fake):
        labels_confuse = self._get_labels(D_out_fake, is_real=True)
        loss_gen = self.criterion(D_out_fake, labels_confuse)

        return loss_gen

    def forward(self, D_out_real=None, D_out_fake=None, mode="D"):
        if mode == "D":
            return self.calculate_D_loss(D_out_real, D_out_fake)
        elif mode == "G":
            return self.calculate_G_loss(D_out_fake)
        else:
            raise ValueError("The mode is not supported")
