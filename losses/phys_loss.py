import torch
import torch.nn as nn 
import torch.nn.functional as F
from .charbonnier_loss import CharbonnierLoss
from .perceptual_loss import PerceptualLoss
from data.utils import restandardize_tensor

# class FM_PhysicalLoss(nn.Module):
#     """
#     Combines:
#     1. Flow Matching Loss (Velocity)
#     2. Physics Consistency Loss (Restoring Input I)
#     3. Perceptual Loss (VGG Content)
#     4. Smoothness Regularization
#     """
#     def __init__(self, loss_config = None):
#         super().__init__()
#         self.charbonnier = CharbonnierLoss()
#         self.perceptual = PerceptualLoss()

#         # --- Default Hyperparameters (Fallbacks) ---
#         # If no config is provided, these defaults are used.
#         self.weights = {
#             "w_flow": 1.0,    # Velocity Matching
#             "w_perc": 0.1,    # VGG Perceptual
#             "w_phys": 0.2,    # Physics Consistency
#             "w_tv": 0.01,     # Transmission Smoothness
#             "w_atm": 0.01     # Atmosphere Constraint
#         }

#         # --- Override with Config ---
#         if loss_config is not None:
#             # We assume loss_config is a dict or an object like cfg.LOSS
#             # We update our weights if the key exists in the config
#             if hasattr(loss_config, "W_FLOW"): self.weights["w_flow"] = loss_config.W_FLOW
#             if hasattr(loss_config, "W_PERC"): self.weights["w_perc"] = loss_config.W_PERC
#             if hasattr(loss_config, "W_PHYS"): self.weights["w_phys"] = loss_config.W_PHYS
#             if hasattr(loss_config, "W_TV"):   self.weights["w_tv"]   = loss_config.W_TV
#             if hasattr(loss_config, "W_ATM"):  self.weights["w_atm"]  = loss_config.W_ATM
            
#             # Support Dictionary access as well (if you pass a raw dict)
#             if isinstance(loss_config, dict):
#                 self.weights.update(loss_config)
                
#     def get_gradients(self, img):
#         dy = img[:, :, 1:, :] - img[:, :, :-1, :]
#         dx = img[:, :, :, 1:] - img[:, :, :, :-1]
#         return dy, dx
    
#     def forward(self, pred_tuple, target_v, x_t, timestep, clean_img, hazy_img):
#         """
#         Args:
#             pred_tuple: (pred_v, t_map, A_pred) from Model
#             target_v:   Ground Truth Velocity (Clean - Hazy)
#             x_t:        Current noisy intermediate image (Normalized [-1, 1])
#             timestep:   Scalar time (B,)
#             clean_img:  Ground Truth Clean Image (Normalized [-1, 1])
#             hazy_img:   Original Hazy Image (Normalized [-1, 1])
#         """
        
#         # Unpack predictions 
#         pred_v, pred_t_map, pred_A = pred_tuple

#         # --- A. Velocity Loss (Keep in Model Space [-1, 1]) --- 
#         # We generally don't un-normalize velocity; MSE on raw logits is fine.
#         loss_v = F.mse_loss(pred_v, target_v) 

#         # --- B. Reconstruction --- 
#         # 1. Get the raw reconstruction in Model Space [-1, 1]
#         t_expand = timestep.view(-1, 1, 1, 1)
#         J_pred_raw = x_t + (1 - t_expand) * pred_v

#         # 2. UN-NORMALIZE EVERYTHING to Image Space [0, 1]
#         # This is where we fix the "Deep Fried" bug using your function.
#         # We must un-normalize the Prediction, the Clean Target, and the Hazy Input
#         # so they are all in the same [0, 1] color space for physics/VGG.
        
#         J_pred_01 = restandardize_tensor(J_pred_raw)
#         clean_img_01 = restandardize_tensor(clean_img)
#         hazy_img_01 = restandardize_tensor(hazy_img)

#         # --- C. Perceptual Loss (Visual Quality) ---
#         # VGG expects [0, 1] inputs.
#         loss_percep = self.perceptual(J_pred_01, clean_img_01)

#         # --- D. Physics Consistency Loss ---
#         # Physics Model: I = J * t + A * (1 - t)
#         # This equation ONLY works if 0=Black and 1=White.
        
#         # Ensure T and A are valid [0, 1] if they aren't already
#         # (Uncomment these if your model outputs raw logits for T/A)
#         # pred_t_map = torch.sigmoid(pred_t_map)
#         # pred_A = torch.sigmoid(pred_A)
        
#         # Re-haze the estimated clean image
#         pred_t_map = torch.clamp(pred_t_map, min=0.01) # Never let it be pure black
#         I_reconstructed = J_pred_01 * pred_t_map + pred_A * (1 - pred_t_map)
        
#         # Compare against the un-normalized hazy image
#         loss_phys = self.charbonnier(I_reconstructed, hazy_img_01)

#         # --- E. Regularizers ---
#         dy, dx = self.get_gradients(pred_t_map)
#         loss_tv = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
        
#         # Penalize A being too dark (< 0.05) or > 1.0 (though sigmoid caps at 1)
#         loss_atm = torch.mean(F.relu(0.05 - pred_A)) 

#         # --- WEIGHTS ---
#         total_loss = (self.weights["w_flow"] * loss_v) + \
#                      (self.weights["w_perc"] * loss_percep) + \
#                      (self.weights["w_phys"] * loss_phys) + \
#                      (self.weights["w_tv"]   * loss_tv) + \
#                      (self.weights["w_atm"]  * loss_atm)

#         return total_loss, {
#             "Total": total_loss.item(),
#             "Flow": loss_v.item(),
#             "VGG": loss_percep.item(),
#             "Phys": loss_phys.item(),
#             "TV": loss_tv.item()
#         }
class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W) images, normalized [0, 1]
        """
        # 1. Compute 2D Fast Fourier Transform (Real-to-Complex)
        # Output shape: (B, C, H, W/2 + 1)
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 2. Compute Loss on Real and Imaginary parts separately
        # This forces the model to learn both Magnitude (Sharpness) and Phase (Structure)
        loss_real = self.criterion(pred_fft.real, target_fft.real)
        loss_imag = self.criterion(pred_fft.imag, target_fft.imag)

        return loss_real + loss_imag


class FM_PhysicalLoss(nn.Module):
    """
    Combines:
    1. Flow Matching Loss (Velocity) - With Density Awareness
    2. Physics Consistency Loss (Restoring Input I)
    3. Perceptual Loss (VGG Content)
    4. Smoothness Regularization
    """
    def __init__(self, loss_config=None):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.perceptual = PerceptualLoss()
        self.fft_loss = FFTLoss()
        
        # Use reduction='none' so we can apply pixel-wise weighting later
        self.mse_none = nn.MSELoss(reduction='none') 

        # --- Default Hyperparameters ---
        self.weights = {
            "w_flow": 1.0,        # Velocity Matching
            "w_perc": 0.2,        # KEEP AT 0.2: Essential for structural integrity
            "w_phys": 0.2,        # Physics Consistency
            "w_fft":  0.1,        # Frequency domain
            "w_tv": 0.01,         # Keeps transmission maps from becoming "noisy"
            "w_atm": 0.01,        # Global atmosphere constraint
            "density_boost": 5.0  # Multiplier for the thickest fog regions
        }

        # --- Override with Config ---
        if loss_config is not None:
            if isinstance(loss_config, dict):
                self.weights.update(loss_config)
            else:
                # Handle YACS CfgNode object
                if hasattr(loss_config, "W_FLOW"): self.weights["w_flow"] = loss_config.W_FLOW
                if hasattr(loss_config, "W_PERC"): self.weights["w_perc"] = loss_config.W_PERC
                if hasattr(loss_config, "W_PHYS"): self.weights["w_phys"] = loss_config.W_PHYS
                if hasattr(loss_config, "W_FFT"):  self.weights["w_fft"]  = loss_config.W_FFT # <--- NEW
                if hasattr(loss_config, "W_TV"):   self.weights["w_tv"]   = loss_config.W_TV
                if hasattr(loss_config, "W_ATM"):  self.weights["w_atm"]  = loss_config.W_ATM
                if hasattr(loss_config, "DENSITY_BOOST"):  self.weights["density_boost"]  = loss_config.DENSITY_BOOST
                
    def get_gradients(self, img):
        """Helper for Total Variation (Smoothness) Loss"""
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dy, dx
    
    def forward(self, pred_tuple, target_v, x_t, timestep, clean_img, hazy_img, current_epoch = None, total_epochs = 100):
        """
        Args:
            pred_tuple: (pred_v, t_map, A_pred) from Model
            target_v:   Ground Truth Velocity (Clean - Hazy)
            x_t:        Current noisy intermediate image (Normalized [-1, 1])
            timestep:   Scalar time (B,)
            clean_img:  Ground Truth Clean Image (Normalized [-1, 1])
            hazy_img:   Original Hazy Image (Normalized [-1, 1])

        We want to add the epochs to schedule the density boost 
            - In the begining, the transmission map needs to learn the basic structure, 
              the model learns the global colors and shapes without being distracted by 
              "hard" spots.
            - The model notices that its "foggy predictions are incurring higher penalties. 
                it starts to sharpening the transmission map to reduce that penalty
            - The model is essentially performing "Hard Example Mining" focusing 
               exclusively on the thickest haze regions where it is struggling 
        """
        
        # Unpack predictions 
        pred_v, pred_t_map, pred_A = pred_tuple

        # --- 1. PREP: Normalize Timestep ---
        # Critical Fix: Ensure timestep is float [0.0, 1.0] for the reconstruction math
        if timestep.max() > 1.0:
            t_norm = timestep.float() / 1000.0
        else:
            t_norm = timestep.float()
        
        t_expand = t_norm.view(-1, 1, 1, 1)

        # --- A. VELOCITY LOSS (Density-Aware) --- 
        # Calculate raw squared error per pixel
        raw_v_loss = self.mse_none(pred_v, target_v)

        # Calculate the Adaptive Boost Scaler
        # Goal: Start with 0 boost (pure MSE) for stability, end with max boost for enhancing detail
        max_boost = self.weights["density_boost"]

        if current_epoch is not None:
            # Normalize progress to [0.0, 1.0]
            progress = current_epoch / float(total_epochs)
            # In case the current epoch is larger than the total epochs (put that for the safety reason)
            progress = max(0.0, min(progress, 1.0))

            # Squared Ramp (x^2)
            # Stays low longer to let the model stabilize, then ramps up 
            adaptive_scalar = progress ** 2
            current_boost = adaptive_scalar * max_boost
        else:
            # We will use the default max_boost if there is no input from the user
            current_boost = max_boost

        
        # Create Weight Map based on Transmission Prediction
        # Low t (Dense Haze) -> High Weight. High t (Clear) -> Low Weight.
        
        # We .detach() t_map so velocity loss doesn't try to "hack" the physics head.
        t_guide = pred_t_map.detach().mean(dim = 1, keepdim = True)

        # This is the formula for the Focus Mechanism.
        # If you think this pixel is thick fog ~ 0.0, pay more attention to that pixel
        # to fix the velocity here 
        pixel_weight = 1.0 + current_boost * (1.0 - t_guide)

        
        # Apply Weight and Mean
        loss_v = (raw_v_loss * pixel_weight).mean()

        # --- B. RECONSTRUCTION (Model Space [-1, 1]) --- 
        # Uses the normalized t_expand to project back to the estimated clean image
        J_pred_raw = x_t + (1 - t_expand) * pred_v

        # --- C. UN-NORMALIZE & SAFETY CLAMP (Image Space [0, 1]) ---
        # 1. Un-normalize standard scaler
        J_pred_01 = restandardize_tensor(J_pred_raw)
        clean_img_01 = restandardize_tensor(clean_img)
        hazy_img_01 = restandardize_tensor(hazy_img)
        
        # 2. Safety Clamp: Ensure VGG never sees exploding values (e.g., -5.0 or 2.0)
        J_pred_safe = torch.clamp(J_pred_01, 0.0, 1.0)

        # --- D. PERCEPTUAL LOSS ---
        # VGG now sees valid [0, 1] images
        loss_percep = self.perceptual(J_pred_safe, clean_img_01)

        # --- E. FFT LOSS (NEW) ---
        # Forces the model to match high-frequency details (textures/edges)
        loss_fft = self.fft_loss(J_pred_safe, clean_img_01)

        # --- F. PHYSICS CONSISTENCY LOSS ---
        # Physics Model: I = J * t + A * (1 - t)
        
        # 1. Clamp Physics Params for Stability
        # Min=0.01: Prevents division by zero or "black hole" gradients
        # Max=1.0:  Prevents "super-white" cheating
        t_map_safe = torch.clamp(pred_t_map, min=0.01, max=1.0)
        pred_A_safe = torch.clamp(pred_A, min=0.0, max=1.0)
        
        # 2. Re-haze the estimated clean image
        I_reconstructed = J_pred_safe * t_map_safe + pred_A_safe * (1 - t_map_safe)
        
        # 3. Compare against original Hazy Image
        loss_phys = self.charbonnier(I_reconstructed, hazy_img_01)

        # --- F. REGULARIZERS ---
        # Smoothness (TV) on the raw transmission map
        dy, dx = self.get_gradients(pred_t_map)
        loss_tv = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
        
        # Atmosphere Constraint: Penalize if A is unrealistically dark (< 0.05)
        loss_atm = torch.mean(F.relu(0.05 - pred_A)) 

        # --- G. TOTAL LOSS ---
        total_loss = (self.weights["w_flow"] * loss_v) + \
                     (self.weights["w_perc"] * loss_percep) + \
                     (self.weights["w_fft"]  * loss_fft) + \
                     (self.weights["w_phys"] * loss_phys) + \
                     (self.weights["w_tv"]   * loss_tv) + \
                     (self.weights["w_atm"]  * loss_atm)

        return total_loss, {
            "Total": total_loss.item(),
            "Flow": loss_v.item(),
            "VGG": loss_percep.item(),
            "FFT": loss_fft.item(),
            "Phys": loss_phys.item(),
            "TV": loss_tv.item()
        }