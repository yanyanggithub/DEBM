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
    Enhanced Time Embedding with better capacity
    """
    def __init__(self, n_out, t_emb_dim=128):
        super().__init__()
        
        # Time Embedding Block with increased capacity
        self.te_block = nn.Sequential(
            nn.Linear(t_emb_dim, n_out * 4),
            nn.SiLU(),
            nn.Linear(n_out * 4, n_out * 2),
            nn.SiLU(),
            nn.Linear(n_out * 2, n_out)
        )
        
        # Improved modulation
        self.modulation = nn.Sequential(
            nn.Linear(n_out, n_out),
            nn.SiLU(),
            nn.Linear(n_out, n_out)
        )
        
    def forward(self, x):
        te = self.te_block(x)
        modulation = self.modulation(te)
        return modulation


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # First conv to change channels if needed
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Main res block with improved convolutions
        self.res_block = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),  # Changed from BatchNorm to GroupNorm
                nn.SiLU()
            ) for _ in range(num_layers)
        ])
        
        # Skip connection with proper channel handling
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Final conv after skip connection
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x, modulation=None):
        # Initial conv
        x = self.conv1(x)
        
        # Main res block with residual connections
        for layer in self.res_block:
            x = layer(x) + x
            
        # Skip connection
        x = x + self.skip(x)
        
        # Modulation if provided
        if modulation is not None:
            x = x * (1 + modulation.view(-1, self.out_channels, 1, 1))
            
        # Final conv
        x = self.conv2(x)
        
        return x


class DoubleConv(nn.Module):
    """
    Enhanced double convolution block with better normalization
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
            nn.GroupNorm(8, out_channels),  # Changed from BatchNorm to GroupNorm
            nn.SiLU(),  # Changed from ReLU to SiLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                      padding=1, bias=False),
            ResBlock(out_channels, out_channels),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsample block with improved pooling
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
    Upsample block with improved interpolation
    """    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                              align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Improved padding for better RGB handling
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, t_emb_dim=128, device="cpu"):
        super().__init__()
        self.n_channels = n_channels
        self.t_emb_dim = t_emb_dim
        self.device = device
        
        # Initial conv with proper RGB handling
        self.inc = DoubleConv(n_channels, 64)
        
        # Down blocks with increased channels
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Up blocks with proper channel handling
        self.up1 = Up(768, 256)  # 512 + 256 = 768 input channels
        self.up2 = Up(384, 128)  # 256 + 128 = 384 input channels
        self.up3 = Up(192, 64)   # 128 + 64 = 192 input channels
        
        # Self attention blocks with improved heads
        self.sa1 = SelfAttention(256, 8)  # Changed from 512 to 256
        self.sa2 = SelfAttention(512, 4)
        self.sa3 = SelfAttention(256, 8)
        
        # Time embedding blocks with proper channel dimensions
        self.time_emb_down1 = TimeEmbedding(128, t_emb_dim)  # 128 channels
        self.time_emb_down2 = TimeEmbedding(256, t_emb_dim)  # 256 channels
        self.time_emb_down3 = TimeEmbedding(512, t_emb_dim)  # 512 channels
        self.time_emb_up1 = TimeEmbedding(256, t_emb_dim)    # 256 channels
        self.time_emb_up2 = TimeEmbedding(128, t_emb_dim)    # 128 channels
        self.time_emb_up3 = TimeEmbedding(64, t_emb_dim)     # 64 channels
        
        # Output conv
        self.outc = OutConv(64, n_channels)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, t):
        """
        Enhanced UNet with improved time embedding integration
        """
        t_emb = get_time_embedding(t, self.t_emb_dim)
        
        # Down path with proper RGB handling
        x1 = self.inc(x)  # 64 channels
        x2 = self.down1(x1)  # 128 channels
        mod1 = self.time_emb_down1(t_emb)
        x2 = x2 * (1 + mod1.view(-1, 128, 1, 1))
        
        x3 = self.down2(x2)  # 256 channels
        mod2 = self.time_emb_down2(t_emb)
        x3 = x3 * (1 + mod2.view(-1, 256, 1, 1))
        x3 = self.sa1(x3)  # 256 channels
        
        x4 = self.down3(x3)  # 512 channels
        mod3 = self.time_emb_down3(t_emb)
        x4 = x4 * (1 + mod3.view(-1, 512, 1, 1))
        x4 = self.sa2(x4)  # 512 channels
        
        # Up path with improved feature handling
        x = self.up1(x4, x3)  # 768 -> 256 channels
        mod4 = self.time_emb_up1(t_emb)
        x = x * (1 + mod4.view(-1, 256, 1, 1))
        x = self.sa3(x)  # 256 channels
        
        x = self.up2(x, x2)  # 384 -> 128 channels
        mod5 = self.time_emb_up2(t_emb)
        x = x * (1 + mod5.view(-1, 128, 1, 1))
        
        x = self.up3(x, x1)  # 192 -> 64 channels
        mod6 = self.time_emb_up3(t_emb)
        x = x * (1 + mod6.view(-1, 64, 1, 1))
        
        return self.outc(x)