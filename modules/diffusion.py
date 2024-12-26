import torch
import torch.nn as nn


class Diffusion(nn.Module):
    def __init__(self, size, channels, timesteps=1000):
        super().__init__()
        self.size = size
        self.channels = channels
        self.timesteps = timesteps

        # Noise schedule
        self.beta_min = 0.0001
        self.beta_max = 0.02
        self.betas = torch.linspace(self.beta_min, self.beta_max, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), 
                                              self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_cumprod)

    def q_sample(self, x_0, t):
        """
        Forward diffusion, sample from q(x_t | x_0)
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
        random_noise = torch.randn_like(x_0)
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * random_noise

    def p_losses(self, model, x_0, t):
        """
        Train the model to predict the noise added at a given timestep.
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t)
        predicted_noise = model(x_t, t)
        return torch.mean((noise - predicted_noise) ** 2)

    def p_sample(self, model, x_t, t):
        """
        Reverse diffusion, sample x_(t-1) from p_theta(x_(t-1) | x_t)
        """
        if t == 0:
            return x_t
        model_mean = self.p_mean_variance(model, x_t, t)[0]
        posterior_variance = self.p_mean_variance(model, x_t, t)[1]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance) * noise

    def p_mean_variance(self, model, x_t, t):
        """
        Compute the mean and variance of the posterior distribution for the 
        reverse diffusion process p_theta(x_(t-1) | x_t)
        """
        model_output = model(x_t, t)
        sqrt_one_minus_alphas_t = torch.sqrt(1.0 - self.alphas[t])
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])

        # Calculate mean
        model_mean = 1.0 / self.sqrt_alphas_bar[t] * (x_t - sqrt_one_minus_alphas_t / self.sqrt_alphas_bar[t] * model_output)

        # Calculate variance
        posterior_variance = self.betas[t] * (1.0 - self.alphas_cumprod[t - 1] / self.alphas_cumprod[t]) * sqrt_recip_alphas_t ** 2

        return model_mean, posterior_variance
