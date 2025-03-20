import torch


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # Adaptive noise schedule based on image type
        self.beta_min = 0.0001  # Base minimum
        self.beta_max = 0.02    # Base maximum
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

        # Image type specific parameters
        self.grayscale_threshold = 0.5  # Threshold for binary features
        self.edge_enhancement = 1.2     # Factor to enhance edge detection
        self.color_enhancement = 1.1    # Factor to enhance color vibrancy

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
        Forward diffusion process with adaptive optimizations for both grayscale and RGB
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        # Proper broadcasting for both grayscale and RGB
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
        
        # Base noise addition
        noisy_x = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
        
        # Channel-specific optimizations
        if x.shape[1] == 1:  # Grayscale
            # Edge enhancement for grayscale
            edges = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            edges = torch.cat([edges, edges[:, :, :, -1:]], dim=3)
            noisy_x = noisy_x + self.edge_enhancement * edges * (1 - sqrt_alpha_bar_t)
        else:  # RGB
            # Color preservation for RGB
            color_diff = torch.abs(x[:, 1:] - x[:, :-1])
            color_diff = torch.cat([color_diff, color_diff[:, -1:]], dim=1)
            noisy_x = noisy_x + self.color_enhancement * color_diff * (1 - sqrt_alpha_bar_t)
        
        return noisy_x
    
    def denoise(self, xt, noise_pred, t):
        """
        Improved reverse diffusion process with adaptive optimizations
        """       
        # Calculate x0 prediction with improved stability
        x0 = xt - (self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise_pred)
        x0 = x0 / self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        
        # Channel-specific processing
        if x0.shape[1] == 1:  # Grayscale
            # Binary feature enhancement for grayscale
            x0 = torch.where(x0 > self.grayscale_threshold, 
                           torch.ones_like(x0), 
                           torch.zeros_like(x0))
        else:  # RGB
            # Color channel normalization for RGB
            x0 = torch.clamp(x0, -1., 1.)
            # Enhance color contrast
            x0 = x0 * self.color_enhancement
            x0 = torch.clamp(x0, -1., 1.)
        
        x0 = torch.clamp(x0, -1., 1.)  # Final clamp for stability
        
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
    