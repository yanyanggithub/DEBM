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
        
        # Base channel dimensions
        base_channels = 32 if n_channels == 1 else 64
        
        # Initial conv with adaptive channel handling
        self.inc = DoubleConv(n_channels, base_channels)
        
        # Down blocks with adaptive channels
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Up blocks with proper channel handling
        self.up1 = Up(base_channels * 12, base_channels * 4)  # 8 + 4 = 12
        self.up2 = Up(base_channels * 6, base_channels * 2)   # 4 + 2 = 6
        self.up3 = Up(base_channels * 3, base_channels)       # 2 + 1 = 3
        
        # Self attention blocks with adaptive heads
        self.sa1 = SelfAttention(base_channels * 4, 4 if n_channels == 1 else 8)
        self.sa2 = SelfAttention(base_channels * 8, 2 if n_channels == 1 else 4)
        self.sa3 = SelfAttention(base_channels * 4, 4 if n_channels == 1 else 8)
        
        # Time embedding blocks with adaptive dimensions
        self.time_emb_down1 = TimeEmbedding(base_channels * 2, t_emb_dim)
        self.time_emb_down2 = TimeEmbedding(base_channels * 4, t_emb_dim)
        self.time_emb_down3 = TimeEmbedding(base_channels * 8, t_emb_dim)
        self.time_emb_up1 = TimeEmbedding(base_channels * 4, t_emb_dim)
        self.time_emb_up2 = TimeEmbedding(base_channels * 2, t_emb_dim)
        self.time_emb_up3 = TimeEmbedding(base_channels, t_emb_dim)
        
        # Output conv
        self.outc = OutConv(base_channels, n_channels)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Adaptive initialization based on input channels
            if self.n_channels == 1:
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            else:
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, t):
        """
        Enhanced UNet with adaptive optimizations for both grayscale and RGB
        """
        # Ensure input has correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        elif len(x.shape) == 5:
            x = x.squeeze(2)  # Remove extra dimension if present
            
        t_emb = get_time_embedding(t, self.t_emb_dim)
        base_channels = 32 if self.n_channels == 1 else 64
        
        # Down path with adaptive channel handling
        x1 = self.inc(x)  # base_channels
        x2 = self.down1(x1)  # base_channels * 2
        mod1 = self.time_emb_down1(t_emb)
        x2 = x2 * (1 + mod1.view(-1, base_channels * 2, 1, 1))
        
        x3 = self.down2(x2)  # base_channels * 4
        mod2 = self.time_emb_down2(t_emb)
        x3 = x3 * (1 + mod2.view(-1, base_channels * 4, 1, 1))
        x3 = self.sa1(x3)  # base_channels * 4
        
        x4 = self.down3(x3)  # base_channels * 8
        mod3 = self.time_emb_down3(t_emb)
        x4 = x4 * (1 + mod3.view(-1, base_channels * 8, 1, 1))
        x4 = self.sa2(x4)  # base_channels * 8
        
        # Up path with adaptive feature handling
        x = self.up1(x4, x3)  # base_channels * 12 -> base_channels * 4
        mod4 = self.time_emb_up1(t_emb)
        x = x * (1 + mod4.view(-1, base_channels * 4, 1, 1))
        x = self.sa3(x)  # base_channels * 4
        
        x = self.up2(x, x2)  # base_channels * 6 -> base_channels * 2
        mod5 = self.time_emb_up2(t_emb)
        x = x * (1 + mod5.view(-1, base_channels * 2, 1, 1))
        
        x = self.up3(x, x1)  # base_channels * 3 -> base_channels
        mod6 = self.time_emb_up3(t_emb)
        x = x * (1 + mod6.view(-1, base_channels, 1, 1))
        
        return self.outc(x)