import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from functools import partial
from tqdm.notebook import tqdm

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def _warmup_beta(linear_start, linear_end, n_timesteps, warmup_frac):
    """
    Creates a beta schedule that linearly increases for a warmup period, 
    then remains constant at the 'linear_end' value.
    """
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Generates a variance (beta) schedule for the diffusion process.
    
    Args:
        schedule (str): The type of schedule ('linear', 'cosine', 'quad', etc.)
        n_timestep (int): Total number of diffusion steps (T).
        linear_start (float): Starting variance at t=0.
        linear_end (float): Ending variance at t=T (used by linear, quad, warmup).
        cosine_s (float): Offset for the cosine schedule to prevent singularities.
        
    Returns:
        betas (np.ndarray or torch.Tensor): 1D array of beta values.
    """
    if schedule == "quad":
        # Quadratic schedule: Starts slow, accelerates towards the end
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2

    elif schedule == "linear":
        # Standard linear schedule (used in original DDPM)
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)

    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)

    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype = np.float64)

    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Jensen-Shannon Divergence schedule: 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
        
    elif schedule == "cosine":
        # Cosine schedule (Nichol and Dhariwal, 2021)
        # Prevents destroying information too quickly in the forward process
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999) # Clamp to prevent numerical instability
    else:
        raise NotImplementedError(schedule)
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(
        self, 
        denoise, 
        image_size, 
        channels = 3,
        loss_type = "l1", 
        conditional = True, 
        schedule_opt = None, 
        freq_weight = 0.01
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise
        self.loss_type = loss_type
        self.conditional = conditional
        self.freq_weight = freq_weight

        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()        

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype = torch.float32, device = device)

        # Assuming make_beta_schedule is defined globally as in your previous snippets
        betas = make_beta_schedule(
            schedule = schedule_opt['schedule'],
            n_timestep = schedule_opt['n_timestep'],
            linear_start = schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas 
        alphas = 1. - betas 
        alphas_cumprod = np.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) # Ensure 1. is a float

        # The core buffers needed for q_sample, q_posterior, and p_mean_variance
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
               self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + \
                         self.posterior_mean_coef2[t] * x_t 
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t] 

        return posterior_mean, posterior_log_variance_clipped
        

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x = None):
        batch_size = x.shape[0]
        
        noise_level = torch.tensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]], dtype = torch.float32
        ).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            noise_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            noise_pred = self.denoise_fn(x, noise_level)

        # Estimate the clean image (x_0)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        # Clipped for stability
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # If we know the current noisy image and we have a good estimate of the clean image
        # The distribution of the previous step x_(t-1) is analytically solvable
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised = True, condition_x = None):
        model_mean, model_log_variance = self.p_mean_variance(
            x = x, t = t, 
            clip_denoised = clip_denoised, 
            condition_x = condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    
    @torch.no_grad()
    def p_sample_loop(self, condition, continuous = False):
        """
        Condition for inputing the pseudo clean image J 
        and also if we set the continuous to True we can return the intermediary values (Debugging)
        """
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        if not self.conditional:
            shape = condition
            img = torch.randn(shape, device=device)
            ret_img = img 
            for i in tqdm(
                reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps
            ):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        else:
            x = condition
            # Generate starting noise with standard 3 channels, even if condition has more
            b, c, h, w = x.shape
            img = torch.randn([b, 3, h, w], device=device)
            ret_img = img

            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        if continuous:
            return img, ret_img
        return img
        
    @torch.no_grad()
    def sample(self, batch_size=1, continuous=False): # Updated continous to continuous
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continuous=continuous)

    
    @torch.no_grad()
    def super_resolution(self, condition, continuous=False):
        # The wrapper used by the DDPM model class during testing
        return self.p_sample_loop(condition, continuous=continuous)        

    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start + \
                (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)
        

    def predict_image(self, x_noisy, continuous_sqrt_alpha_cumprod, noise=None):
        """Used during training to predict x_0 for the Frequency Loss."""
        return ((x_noisy - (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise) / continuous_sqrt_alpha_cumprod)

    
    def p_losses(self, x_in, condition, noise = None):
        x_start = x_in
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)

        # Fixed syntax error: comma instead of dot before .to(device)
        continuous_sqrt_alpha_cumprod = torch.tensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size = b
            ),
            dtype = torch.float32 
        ).to(x_start.device)
        
        # Fixed typo: comprod -> cumprod
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))

        # 1. Forward processes (Add noise)
        x_noisy = self.q_sample(
            x_start=x_start, 
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
            noise=noise
        )

        # 2. Predict the noise
        if not self.conditional: 
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(torch.cat([condition, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # 3. Standard Spatial Noise Loss
        loss_spatial = self.loss_func(noise, x_recon)

        
        
        # 4. Frequency Prior Optimization Loss
        # Predict the clean image from the current noisy state and predicted noise
        x0_recon = self.predict_image(
            x_noisy=x_noisy, 
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
            noise=x_recon
        )

        # Fast Fourier Transform (FFT) on predicted x_0 and ground truth x_in
        x0_recon_fft = torch.abs(torch.fft.fftn(x0_recon, dim = (-2, -1)))
        x_in_fft = torch.abs(torch.fft.fftn(x_in, dim=(-2, -1)))

        loss_frequency = self.loss_func(x0_recon_fft, x_in_fft)
        
        return loss_spatial + (self.freq_weight * loss_frequency)

    def forward(self, x, condition, *args, **kwargs):
        return self.p_losses(x, condition, *args, **kwargs)