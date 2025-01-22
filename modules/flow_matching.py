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