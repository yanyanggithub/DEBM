import torch
from torch import nn
from torch.nn import functional as F


class RBMConfig:
    n_visible:int = 784 
    n_hidden:int = 128 
    k:int = 1


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine
    """
    def __init__(self, n_visible, n_hidden, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(1, n_visible))
        self.h_bias = nn.Parameter(torch.zeros(1, n_hidden))
        self.k  = k
    
    def _sample(self, prob):
        return torch.distributions.Bernoulli(prob).sample()

    def _pass(self, v):
        prob_h = torch.sigmoid(F.linear(v, self.weight, self.h_bias))
        return self._sample(prob_h)
    
    def _reverse_pass(self, h):
        prob_v = torch.sigmoid(F.linear(h, self.weight.t(), self.v_bias))
        return prob_v
    
    def energy(self, v):
        t1 = -torch.matmul(v, self.v_bias.t())
        h_hat = F.linear(v, self.weight, self.h_bias)
        t2 = -torch.sum(F.softplus(h_hat), dim=1)
        return torch.mean(t1 + t2)
    
    def forward(self, v):
        h = self._pass(v)
        for _ in range(self.k):
            v_reconstructed = self._reverse_pass(h)
            h = self._pass(v_reconstructed)
        return v, v_reconstructed

