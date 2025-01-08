import torch


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu"):
        super().__init__()
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
        x0 = xt - (torch.sqrt(1 - self.alpha_bars[t]) * noise_pred)
        x0 = x0/torch.sqrt(self.alpha_bars[t])
        x0 = torch.clamp(x0, -1., 1.) 
        
        # mean of x_(t-1)
        mean = (xt - ((1 - self.alphas[t]) * noise_pred)
                /(torch.sqrt(1 - self.alpha_bars[t])))
        mean = mean/(torch.sqrt(self.alphas[t]))
        
        if t == 0:
            return mean, x0        
        else:
            variance =  (1 - self.alpha_bars[t-1])/(1 - self.alpha_bars[t])
            variance = variance * self.betas[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(self.device)
            
        return mean + sigma * z, x0
    