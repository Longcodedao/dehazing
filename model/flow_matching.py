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