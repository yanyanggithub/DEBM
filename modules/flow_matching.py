import torch


class FlowMatching:
    def __init__(self, sigma=0.1, device="cpu"):
        self.sigma = sigma
        self.device = device

    def compute_flow(self, x0, x1, t):
        """
        flow interpolation at time t
        """
        return t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
    
    def sample_xt(self, x0, x1, t):
        """
        sample from the distribution p(x_t|x_0, x_1)
        """
        flow = self.compute_flow(x0, x1, t)
        noise = torch.randn_like(x0).to(self.device)
        xt = flow  + self.sigma * noise
        conditional_flow = x1 - x0
        return xt, conditional_flow