import torch
import torch.nn as nn 
import torch.nn.functional as F
from .charbonnier_loss import CharbonnierLoss
from .perceptual_loss import PerceptualLoss
from data.utils import restandardize_tensor

class FM_PhysicalLoss(nn.Module):
    """
    Combines:
    1. Flow Matching Loss (Velocity)
    2. Physics Consistency Loss (Restoring Input I)
    3. Perceptual Loss (VGG Content)
    4. Smoothness Regularization
    """
    def __init__(self, loss_config = None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.perceptual = PerceptualLoss()

        # --- Default Hyperparameters (Fallbacks) ---
        # If no config is provided, these defaults are used.
        self.weights = {
            "w_flow": 1.0,    # Velocity Matching
            "w_perc": 0.1,    # VGG Perceptual
            "w_phys": 0.2,    # Physics Consistency
            "w_tv": 0.01,     # Transmission Smoothness
            "w_atm": 0.01     # Atmosphere Constraint
        }

        # --- Override with Config ---
        if loss_config is not None:
            # We assume loss_config is a dict or an object like cfg.LOSS
            # We update our weights if the key exists in the config
            if hasattr(loss_config, "W_FLOW"): self.weights["w_flow"] = loss_config.W_FLOW
            if hasattr(loss_config, "W_PERC"): self.weights["w_perc"] = loss_config.W_PERC
            if hasattr(loss_config, "W_PHYS"): self.weights["w_phys"] = loss_config.W_PHYS
            if hasattr(loss_config, "W_TV"):   self.weights["w_tv"]   = loss_config.W_TV
            if hasattr(loss_config, "W_ATM"):  self.weights["w_atm"]  = loss_config.W_ATM
            
            # Support Dictionary access as well (if you pass a raw dict)
            if isinstance(loss_config, dict):
                self.weights.update(loss_config)
                
    def get_gradients(self, img):
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dy, dx
    
    def forward(self, pred_tuple, target_v, x_t, timestep, clean_img, hazy_img):
        """
        Args:
            pred_tuple: (pred_v, t_map, A_pred) from Model
            target_v:   Ground Truth Velocity (Clean - Hazy)
            x_t:        Current noisy intermediate image (Normalized [-1, 1])
            timestep:   Scalar time (B,)
            clean_img:  Ground Truth Clean Image (Normalized [-1, 1])
            hazy_img:   Original Hazy Image (Normalized [-1, 1])
        """
        
        # Unpack predictions 
        pred_v, pred_t_map, pred_A = pred_tuple

        # --- A. Velocity Loss (Keep in Model Space [-1, 1]) --- 
        # We generally don't un-normalize velocity; MSE on raw logits is fine.
        loss_v = F.mse_loss(pred_v, target_v) 

        # --- B. Reconstruction --- 
        # 1. Get the raw reconstruction in Model Space [-1, 1]
        t_expand = timestep.view(-1, 1, 1, 1)
        J_pred_raw = x_t + (1 - t_expand) * pred_v

        # 2. UN-NORMALIZE EVERYTHING to Image Space [0, 1]
        # This is where we fix the "Deep Fried" bug using your function.
        # We must un-normalize the Prediction, the Clean Target, and the Hazy Input
        # so they are all in the same [0, 1] color space for physics/VGG.
        
        J_pred_01 = restandardize_tensor(J_pred_raw)
        clean_img_01 = restandardize_tensor(clean_img)
        hazy_img_01 = restandardize_tensor(hazy_img)

        # --- C. Perceptual Loss (Visual Quality) ---
        # VGG expects [0, 1] inputs.
        loss_percep = self.perceptual(J_pred_01, clean_img_01)

        # --- D. Physics Consistency Loss ---
        # Physics Model: I = J * t + A * (1 - t)
        # This equation ONLY works if 0=Black and 1=White.
        
        # Ensure T and A are valid [0, 1] if they aren't already
        # (Uncomment these if your model outputs raw logits for T/A)
        # pred_t_map = torch.sigmoid(pred_t_map)
        # pred_A = torch.sigmoid(pred_A)
        
        # Re-haze the estimated clean image
        I_reconstructed = J_pred_01 * pred_t_map + pred_A * (1 - pred_t_map)
        
        # Compare against the un-normalized hazy image
        loss_phys = self.charbonnier(I_reconstructed, hazy_img_01)

        # --- E. Regularizers ---
        dy, dx = self.get_gradients(pred_t_map)
        loss_tv = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
        
        # Penalize A being too dark (< 0.05) or > 1.0 (though sigmoid caps at 1)
        loss_atm = torch.mean(F.relu(0.05 - pred_A)) 

        # --- WEIGHTS ---
        total_loss = (self.weights["w_flow"] * loss_v) + \
                     (self.weights["w_perc"] * loss_percep) + \
                     (self.weights["w_phys"] * loss_phys) + \
                     (self.weights["w_tv"]   * loss_tv) + \
                     (self.weights["w_atm"]  * loss_atm)

        return total_loss, {
            "Total": total_loss.item(),
            "Flow": loss_v.item(),
            "VGG": loss_percep.item(),
            "Phys": loss_phys.item(),
            "TV": loss_tv.item()
        }