import torch
import torch.nn as nn 
import torch.nn.functional as F
from .charbonnier_loss import CharbonnierLoss
from .perceptual_loss import PerceptualLoss, ContrastiveLoss
from data.utils import restandardize_tensor


class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 1. Magnitude Loss (Amplitude/Style)
        # L1 is perfect here because magnitude is linear (0 to infinity)
        loss_mag = self.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        # 2. Phase Loss (Structure/Edges)
        # Cosine distance handles the -pi to pi wrapS-around correctly
        # Use this for safety :))) 
        pred_angle = torch.angle(pred_fft)
        target_angle = torch.angle(target_fft)
        loss_pha = torch.mean(1 - torch.cos(pred_angle - target_angle))

        return 0.5 * loss_mag + 0.5 * loss_pha


# --- 3. HELPER: SSIM (Metric Booster) ---
# Simple implementation of SSIM for loss
def ssim_loss(img1, img2):
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1**2, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()

        
class FM_PhysicalLoss(nn.Module):
    def __init__(self, loss_config=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        
        # We need to modify PerceptualLoss to expose feature extraction
        self.perceptual = PerceptualLoss() 
        self.fft_loss = FFTLoss()
        self.mse_none = nn.MSELoss(reduction='none') 
        
        # New: Contrastive
        self.contrastive = ContrastiveLoss(self.perceptual)

        self.weights = {
            "w_flow": 1.0,        
            "w_perc": 0.2,       
            "w_phys": 0.2,       
            "w_fft":  0.1,       
            "w_tv": 0.01,         
            "w_atm": 0.01,
            "w_cr": 0.1,         # New: Contrastive Weight
            "w_ssim": 0.2,       # New: SSIM Weight
            "density_boost": 5.0 
        }
        
        # (Config update logic remains the same...)
        if loss_config is not None:
             # ... (your existing config code) ...
             if hasattr(loss_config, "W_CR"): self.weights["w_cr"] = loss_config.W_CR

    def get_gradients(self, img):
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dy, dx
    
    def forward(self, pred_tuple, target_v, x_t, timestep, clean_img, hazy_img, current_epoch=None, total_epochs=100):
        pred_v, pred_t_map, pred_A = pred_tuple

        # --- A. TIME NORM ---
        if timestep.max() > 1.0:
            t_norm = timestep.float() / 1000.0
        else:
            t_norm = timestep.float()
        t_expand = t_norm.view(-1, 1, 1, 1)

        # --- B. DENSITY-AWARE FLOW LOSS ---
        raw_v_loss = self.mse_none(pred_v, target_v)
        
        max_boost = self.weights["density_boost"]
        if current_epoch is not None:
            progress = max(0.0, min(current_epoch / float(total_epochs), 1.0))
            adaptive_scalar = progress ** 2
            current_boost = adaptive_scalar * max_boost 
        else:
            current_boost = max_boost
        
        t_guide = pred_t_map.detach().mean(dim=1, keepdim=True)
        pixel_weight = 1.0 + current_boost * (1.0 - t_guide)
        loss_v = (raw_v_loss * pixel_weight).mean()

        # --- C. RECONSTRUCTION ---
        J_pred_raw = x_t + (1 - t_expand) * pred_v
        
        J_pred_01 = restandardize_tensor(J_pred_raw)
        clean_img_01 = restandardize_tensor(clean_img)
        hazy_img_01 = restandardize_tensor(hazy_img)
        
        J_pred_safe = torch.clamp(J_pred_01, 0.0, 1.0)

        # --- D. STANDARD LOSSES ---
        loss_percep = self.perceptual(J_pred_safe, clean_img_01)
        loss_fft = self.fft_loss(J_pred_safe, clean_img_01)
        
        # --- E. NEW LOSSES ---
        # 1. SSIM Loss (Directly maximizes Metric)
        loss_ssim = ssim_loss(J_pred_safe, clean_img_01)
        
        # 2. Contrastive Loss (Push away from Hazy)
        # Note: We pass hazy_img_01 as the "Negative"
        loss_cr = self.contrastive(J_pred_safe, clean_img_01, hazy_img_01)

        # --- F. PHYSICS CONSISTENCY ---
        t_map_safe = torch.clamp(pred_t_map, min=0.01, max=1.0)
        pred_A_safe = torch.clamp(pred_A, min=0.0, max=1.0)
        
        I_reconstructed = J_pred_safe * t_map_safe + pred_A_safe * (1 - t_map_safe)
        loss_phys = self.charbonnier(I_reconstructed, hazy_img_01)

        # --- G. REGULARIZERS ---
        dy, dx = self.get_gradients(pred_t_map)
        loss_tv = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
        loss_atm = torch.mean(F.relu(0.05 - pred_A)) 

        # --- H. AGGREGATION ---
        total_loss = (self.weights["w_flow"] * loss_v) + \
                     (self.weights["w_perc"] * loss_percep) + \
                     (self.weights["w_fft"]  * loss_fft) + \
                     (self.weights["w_phys"] * loss_phys) + \
                     (self.weights["w_tv"]   * loss_tv) + \
                     (self.weights["w_atm"]  * loss_atm) + \
                     (self.weights["w_cr"]   * loss_cr) + \
                     (self.weights["w_ssim"] * loss_ssim)

        return total_loss, {
            "Total": total_loss.item(),
            "Flow": loss_v.item(),
            "VGG": loss_percep.item(),
            "CR": loss_cr.item(),   # Track this!
            "SSIM": loss_ssim.item(),
            "FFT": loss_fft.item(),
            "Phys": loss_phys.item()
        }