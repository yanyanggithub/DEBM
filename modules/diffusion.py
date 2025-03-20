import torch


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # Improved noise schedule following Stable Diffusion
        self.beta_min = 0.00085  # Adjusted for better quality
        self.beta_max = 0.0120   # Adjusted for better quality
        self.betas = torch.linspace(self.beta_min, 
                                    self.beta_max, self.timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        
        # Pre-compute coefficients for faster sampling
        self.coef1 = torch.sqrt(1 / self.alphas)
        self.coef2 = self.coef1 * (1 - self.alphas) / self.sqrt_one_minus_alpha_bars
        self.coef3 = torch.sqrt(self.betas)

    def reset(self):
        self.betas = torch.linspace(self.beta_min, 
                                    self.beta_max, 
                                    self.timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        
        # Reset pre-computed coefficients
        self.coef1 = torch.sqrt(1 / self.alphas)
        self.coef2 = self.coef1 * (1 - self.alphas) / self.sqrt_one_minus_alpha_bars
        self.coef3 = torch.sqrt(self.betas)

    def add_noise(self, x, noise, t):
        """
        Forward diffusion process with improved broadcasting
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        # Proper broadcasting for RGB images
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    
    def denoise(self, xt, noise_pred, t):
        """
        Improved reverse diffusion process with better sampling
        """       
        # Calculate x0 prediction with improved stability
        x0 = xt - (self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise_pred)
        x0 = x0 / self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        x0 = torch.clamp(x0, -1., 1.)  # Clamp to [-1,1] for better stability
        
        # Calculate mean with pre-computed coefficients
        mean = self.coef1[t].view(-1, 1, 1, 1) * (
            xt - self.coef2[t].view(-1, 1, 1, 1) * noise_pred
        )
        mean = torch.clamp(mean, -1., 1.)
        
        if t == 0:
            # Normalize to [0,1] range for final output
            mean = (mean + 1) / 2
            x0 = (x0 + 1) / 2
            return mean, x0
        
        # Calculate variance with improved stability
        variance = self.betas[t] * (1 - self.alpha_bars[t-1])/(1 - self.alpha_bars[t])
        sigma = self.coef3[t]
        z = torch.randn(xt.shape).to(self.device)
        
        # Sample with improved stability
        sample = mean + sigma.view(-1, 1, 1, 1) * z
        sample = torch.clamp(sample, -1., 1.)
        
        return sample, x0
    