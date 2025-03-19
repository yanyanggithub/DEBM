import torch


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.beta_min = 0.0001
        self.beta_max = 0.02
        self.betas = torch.linspace(self.beta_min, 
                                    self.beta_max, self.timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)        

    def reset(self):
        self.betas = torch.linspace(self.beta_min, 
                                    self.beta_max, 
                                    self.timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)        

    def add_noise(self, x, noise, t):
        """
        Forward diffusion process
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        # Broadcast to multiply with the original image.
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    
    def denoise(self, xt, noise_pred, t):
        """
        reverse diffusion process 
        sample from the distribution p(x_(t-1)|x_t)
        """       
        # Calculate x0 prediction
        x0 = xt - (torch.sqrt(1 - self.alpha_bars[t]) * noise_pred)
        x0 = x0/torch.sqrt(self.alpha_bars[t])
        x0 = torch.clamp(x0, 0., 1.)  # Clamp to [0,1] for image data
        
        # Calculate mean of x_(t-1)
        mean = (1/torch.sqrt(self.alphas[t])) * (
            xt - ((1 - self.alphas[t])/torch.sqrt(1 - self.alpha_bars[t])) * noise_pred
        )
        mean = torch.clamp(mean, 0., 1.)  # Clamp mean to [0,1]
        
        if t == 0:
            return mean, x0
        
        # Calculate variance for t > 0
        variance = self.betas[t] * (1 - self.alpha_bars[t-1])/(1 - self.alpha_bars[t])
        sigma = torch.sqrt(variance)
        z = torch.randn(xt.shape).to(self.device)
        
        # Sample and clamp the final result
        sample = mean + sigma * z
        sample = torch.clamp(sample, 0., 1.)
        
        return sample, x0
    