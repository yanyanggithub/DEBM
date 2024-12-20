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
        return torch.bernoulli(prob)

    def _pass(self, v):
        h_prob = torch.sigmoid(F.linear(v, self.weight, self.h_bias))
        return h_prob, self._sample(h_prob)
    
    def _reverse_pass(self, h):
        v_prob = torch.sigmoid(F.linear(h, self.weight.t(), self.v_bias))
        return v_prob, self._sample(v_prob)
    
    def constrastive_divergence(self, X, lr=0.01, batch_size=64):
        pos_h_prob, pos_h_sample = self._pass(X)
        pos_gradient = torch.matmul(pos_h_sample.t(), X)

        h_sample = pos_h_sample
        for _ in range(self.k):
            v_recon_prob, _ = self._reverse_pass(h_sample)
            h_prob, h_sample = self._pass(v_recon_prob)

        neg_gradient = torch.matmul(h_prob.t(), v_recon_prob)
        gradient = pos_gradient - neg_gradient
        gradient = gradient/batch_size

        dv_bias = torch.sum(X - v_recon_prob, dim=0)/batch_size
        dh_bias = torch.sum(pos_h_prob - h_prob, dim=0)/batch_size
        with torch.no_grad():
            self.weight += lr * gradient
            self.v_bias += lr * dv_bias
            self.h_bias += lr * dh_bias

        loss = torch.mean(torch.sum((X - v_recon_prob)**2, dim=0))
        return loss
    
    def forward(self, v):
        _, h_sample = self._pass(v)
        for _ in range(self.k):
            v_reconstructed, _ = self._reverse_pass(h_sample)
            h_prob, h_sample = self._pass(v_reconstructed)
        return h_prob, v_reconstructed

