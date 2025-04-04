import torch

class Flow:
    def __init__(self, device="cpu"):
        self.device = device

    def interpolate(self, x0, x1, t):
        """
        flow interpolation at time t
        """
        return t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
    
    def sample_xt(self, x0, x1, t):
        """
        sample xt and flow
        """
        xt = self.interpolate(x0, x1, t)
        flow = x1 - x0
        return xt, flow

    def get_t_steps(self, num_steps):
        """
        Compute time steps for num_steps, with t_0=0 and t_T=1.
        """
        return torch.linspace(0, 1, num_steps + 1)[1:-1].to(self.device)