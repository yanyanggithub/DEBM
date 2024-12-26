import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim # Dimension of the input embeddings
        self.heads = heads # Number of attention heads.
        self.head_dim = embed_dim // heads # Dimension of each attention head

        assert (
            self.head_dim * heads == embed_dim
        ), "Embed dim must be divisible by heads"

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Split into multiple heads
        q = self.wq(x).reshape(batch_size, seq_len, self.heads, self.head_dim)
        k = self.wk(x).reshape(batch_size, seq_len, self.heads, self.head_dim)
        v = self.wv(x).reshape(batch_size, seq_len, self.heads, self.head_dim)

        # Calculate attention scores
        attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) / torch.sqrt(
            torch.tensor(self.head_dim).float()
        )

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Calculate weighted sum of values
        out = torch.einsum("bhij,bhjd->bhid", attention_weights, v)

        # Concatenate heads and apply final linear layer
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        out = self.fc_out(out)

        return out


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, heads):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim # Dimension of the query embeddings
        self.key_value_dim = key_value_dim # Dimension of the key and value embeddings
        self.heads = heads # Number of attention heads
        self.head_dim = key_value_dim // heads # Dimension of each attention head

        assert (
            self.head_dim * heads == key_value_dim
        ), "Key/Value dim must be divisible by heads"

        self.wq = nn.Linear(query_dim, key_value_dim)
        self.wk = nn.Linear(key_value_dim, key_value_dim)
        self.wv = nn.Linear(key_value_dim, key_value_dim)
        self.fc_out = nn.Linear(key_value_dim, query_dim)

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Split into multiple heads
        q = self.wq(query).reshape(batch_size, query_len, self.heads, self.head_dim)
        k = self.wk(key).reshape(batch_size, key_len, self.heads, self.head_dim)
        v = self.wv(value).reshape(batch_size, key_len, self.heads, self.head_dim)

        # Calculate attention scores
        attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) / torch.sqrt(
            torch.tensor(self.head_dim).float()
        )

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Calculate weighted sum of values
        out = torch.einsum("bhij,bhjd->bhid", attention_weights, v)

        # Concatenate heads and apply final linear layer
        out = out.reshape(batch_size, query_len, self.key_value_dim)
        out = self.fc_out(out)

        return out