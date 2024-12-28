import math
import torch
import torch.nn as nn
from .unet import *
from .attention import SelfAttention


def pos_encoding(t, n_channels, embed_size):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, n_channels, 2).float() / n_channels)
    )
    pos_enc_a = torch.sin(t.repeat(1, n_channels // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, n_channels // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc.view(-1, n_channels, 1, 1).repeat(1, 1, embed_size, embed_size)


class Diffusion(nn.Module):
    def __init__(self, input_size, n_channels, timesteps=1000):
        super().__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        self.timesteps = timesteps

        self.beta_min = 0.0001
        self.beta_max = 0.02

        # unet
        bilinear = True
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)
        self.sa1 = SelfAttention(256, 8)
        self.sa2 = SelfAttention(256, 4)
        self.sa3 = SelfAttention(128, 8)

    def add_noise(self, x, t):
        """
        Forward diffusion process
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        noise = torch.randn_like(x)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    
    @torch.no_grad()
    def denoise(self, x, t):
        """
        inner loop of Algorithm 2 from (Ho et al., 2020).
        """       
        if t > 1:
            z = torch.randn(x.shape)
        else:
            z = 0
        e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
        pre_scale = 1 / math.sqrt(self.alpha(t))
        e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
        post_sigma = math.sqrt(self.beta(t)) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma
        return x
    
    def beta(self, t):
        return self.beta_min + (t / self.timesteps) * (
            self.beta_max - self.beta_min
        )
    
    def alpha(self, t):
        return 1 - self.beta(t)
    
    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def forward(self, x, t):
        """
        unet + self attention
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + pos_encoding(t, 128, 14)
        x3 = self.down2(x2) + pos_encoding(t, 256, 7)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + pos_encoding(t, 256, 3)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + pos_encoding(t, 128, 7)
        x = self.sa3(x)
        x = self.up2(x, x2) + pos_encoding(t, 64, 14)
        x = self.up3(x, x1) + pos_encoding(t, 64, 28)
        output = self.outc(x)
        return output

    def fit(self, batch):
        """
        Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.timesteps, [batch.shape[0]])
        noise_imgs = []
        epsilons = torch.randn(batch.shape)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + 
                (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.input_size), 
            epsilons.reshape(-1, self.input_size)
        )
        # self.log("train/loss", loss)
        return loss