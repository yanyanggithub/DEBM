import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, n_channels, n_heads):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        
        # Improved attention with separate Q, K, V projections
        self.q_proj = nn.Linear(n_channels, n_channels)
        self.k_proj = nn.Linear(n_channels, n_channels)
        self.v_proj = nn.Linear(n_channels, n_channels)
        self.out_proj = nn.Linear(n_channels, n_channels)
        
        # Adaptive normalization based on channel count
        self.g_norm = nn.GroupNorm(4 if n_channels <= 64 else 8, n_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Image type specific parameters
        self.is_grayscale = n_channels <= 64
        self.edge_weight = 1.5 if self.is_grayscale else 1.0
        self.color_weight = 1.2 if not self.is_grayscale else 1.0

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        
        # Reshape for attention
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)  # [B, H*W, C]
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, H*W, C]
        k = self.k_proj(x)  # [B, H*W, C]
        v = self.v_proj(x)  # [B, H*W, C]
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, h*w, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, h*w, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, h*w, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute base attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Image type specific attention enhancements
        if self.is_grayscale:
            # Edge-aware attention for grayscale
            edge_attn = torch.abs(q - k.transpose(-2, -1))
            attn = attn + self.edge_weight * edge_attn
        else:
            # Color-aware attention for RGB
            # Compute color differences in the feature space
            q_color = q.reshape(batch_size, self.n_heads, h*w, self.head_dim)
            k_color = k.transpose(-2, -1).reshape(batch_size, self.n_heads, h*w, self.head_dim)
            color_attn = torch.abs(q_color - k_color).mean(dim=-1, keepdim=True)
            attn = attn + self.color_weight * color_attn
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, h*w, channels)
        
        # Project output
        out = self.out_proj(out)
        
        # Reshape back to image
        out = out.transpose(1, 2).reshape(batch_size, channels, h, w)
        
        return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, heads):
        super().__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.heads = heads
        self.head_dim = key_value_dim // heads

        assert (
            self.head_dim * heads == key_value_dim
        ), "Key/Value dim must be divisible by heads"

        # Improved projections with separate Q, K, V
        self.q_proj = nn.Linear(query_dim, key_value_dim)
        self.k_proj = nn.Linear(key_value_dim, key_value_dim)
        self.v_proj = nn.Linear(key_value_dim, key_value_dim)
        self.out_proj = nn.Linear(key_value_dim, query_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Scale factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query)  # [B, Q, C]
        k = self.k_proj(key)    # [B, K, C]
        v = self.v_proj(value)  # [B, K, C]

        # Reshape for multi-head attention
        q = q.reshape(batch_size, query_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, key_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, key_len, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, query_len, self.key_value_dim)
        
        # Project output
        out = self.out_proj(out)

        return out