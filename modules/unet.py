import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention


def get_time_embedding(timesteps, t_emb_dim):
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start = 0, 
                              end = t_emb_dim//2, 
                              dtype=torch.float32, 
                              device=timesteps.device
                             ) / (t_emb_dim)
    
    factor = 10000**factor

    t_emb = timesteps[:,None] # B -> (B, 1) 
    t_emb = t_emb/factor # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1) # (B , t_emb_dim)
    
    return t_emb


class TimeEmbedding(nn.Module):
    """
    Maps the Time Embedding to the Required output Dimension.
    """
    def __init__(self, n_out, t_emb_dim=128):
        super().__init__()
        
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )
        
    def forward(self, x):
        return self.te_block(x)
    

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsample block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """
    Upsample block
    """    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                              align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2's dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, t_emb_dim=128, device="cpu"):
        super().__init__()
        self.n_channels = n_channels
        self.t_emb_dim = t_emb_dim
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.sa1 = SelfAttention(256, 8)
        self.sa2 = SelfAttention(256, 4)
        self.sa3 = SelfAttention(128, 8)
        self.time_emb_down1 = TimeEmbedding(128, t_emb_dim)       
        self.time_emb_down2 = TimeEmbedding(256, t_emb_dim)
        self.time_emb_down3 = TimeEmbedding(256, t_emb_dim)
        self.time_emb_up1 = TimeEmbedding(128, t_emb_dim)
        self.time_emb_up2 = TimeEmbedding(64, t_emb_dim)
        self.time_emb_up3 = TimeEmbedding(64, t_emb_dim)
        self.outc = OutConv(64, n_channels)
        self.device = device    
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, t):
        """
        unet + self attention
        """
        t_emb = get_time_embedding(t, self.t_emb_dim)
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.time_emb_down1(t_emb)[:, :, None, None]
        x3 = self.down2(x2) + self.time_emb_down2(t_emb)[:, :, None, None]
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.time_emb_down3(t_emb)[:, :, None, None]
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.time_emb_up1(t_emb)[:, :, None, None]
        x = self.sa3(x)
        x = self.up2(x, x2) + self.time_emb_up2(t_emb)[:, :, None, None]
        x = self.up3(x, x1) + self.time_emb_up3(t_emb)[:, :, None, None]
        output = self.outc(x)
        return output