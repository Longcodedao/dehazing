import torch
from torchdiffeq import odeint

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
    def __init__(self, model):
        self.model = model
        
    def ode_func(self, t, x):
        t = t.expand(x.size(0))
        # Model returns (v, t_map, A), we only need v for integration
        v_pred, _, _ = self.model(x, t)
        return v_pred

    @torch.no_grad()
    def sample(self, x_init, nfe = 20):
        t_span = torch.linspace(0, 1, nfe, device=x_init.device)
        
        solution = odeint(
            self.ode_func, x_init, t_span, rtol=1e-4, atol=1e-4, method="euler"
        )
        return solution[-1]
