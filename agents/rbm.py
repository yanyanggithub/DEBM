import torch
from torch import nn
from torch.nn import functional as F


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k  = k
    
    def _pass(self, v):
        prob_h = torch.sigmoid(F.linear(v, self.weight, self.h_bias))
        h_sample = prob_h.bernoulli()
        return h_sample
    
    def _reverse_pass(self, h):
        prob_v = torch.sigmoid(F.linear(h, self.weight.t(), self.v_bias))
        v_sample = prob_v.bernoulli()
        return v_sample
    
    def energy(self, v):
        v1 = torch.matmul(v, self.v_bias)
        v2 = F.linear(v, self.weight, self.h_bias)
        h1 = torch.sum(torch.log(1 + torch.exp(v2)), dim=1)
        return -(v1 + h1)
    
    def forward(self, v):
        h = self._pass(v)
        for _ in range(self.k):
            v_sample = self._reverse_pass(h)
            h = self._pass(v_sample)
        return v_sample

