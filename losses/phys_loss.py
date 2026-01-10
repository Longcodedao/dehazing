import torch
import torch.nn as nn 
import torch.nn.functional as F
from .charbonnier_loss import CharbonnierLoss
from .perceptual_loss import PerceptualLoss

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
            x_t:        Current noisy intermediate image
            timestep:   Scalar time (B,) for Flow Matching
            clean_img:  Ground Truth Clean Image (Target J)
            hazy_img:   Original Hazy Image (Input I) - REQUIRED for Phys Loss
        """
        
        # Unpack predictions 
        pred_v, pred_t_map, pred_A = pred_tuple

        # --- A. Velocity Loss (Flow Matching Core) --- 
        # "Learn to move pixels from Hazy to Clean"
        loss_v = F.mse_loss(pred_v, target_v) 

        # --- B. RECONSTRUCTED IMAGE -- 
        # 2. Reconstruct Estimated Clean Image for Perceptual Loss
        # Formula: x_1 = x_t + (1 - t) * v_pred
        # We need this because VGG expects an IMAGE, not a velocity vector.
        t_expand = timestep.view(-1, 1, 1, 1)
        J_pred = x_t + (1 - t_expand) * pred_v

        # Clamp to ensure stability for VGG and Physics
        J_pred = torch.clamp(J_pred, 0.0, 1.0)

        # --- C. PERCEPTUAL LOSS (Visual Quality) ---
        # "Make the reconstructed image J look like a natural image"
        loss_percep = self.perceptual(J_pred, clean_img)

        # --- D. Physics Consistency Loss ---
        # "If we re-haze our estimated J using our predicted T and A, 
        # do we get back the original hazy image?"
        # Physics Model: I = J * t + A * (1 - t)
        
        # Ensure T and A are in valid ranges if model doesn't enforce it
        # (Assuming model output is already Sigmoid-ed, otherwise uncomment below)
        # pred_t_map = torch.sigmoid(pred_t_map)
        # pred_A = torch.sigmoid(pred_A)
        I_reconstructed = J_pred * pred_t_map  + pred_A * (1 - pred_t_map)
        loss_phys = self.charbonnier(I_reconstructed, hazy_img)

        # --- E. REGULARIZERS ---
        # 1. TV Loss: Smoothness for Transmission map
        dy, dx = self.get_gradients(pred_t_map)
        loss_tv = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
        
        # 2. Atmosphere: Prevent impossible A values
        # We penalize A being too dark (< 0.1) or oversaturated (> 1.0)
        # Your previous (0.5 - A) constraint forces A > 0.5, which is risky for night scenes.
        # This is safer:        
        loss_atm = torch.mean(F.relu(0.5 - pred_A))

        # --- WEIGHTS ---
        # Adjust these if specific parts are failing
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