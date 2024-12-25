import torch.nn as nn
from .rbm import RBM


class StackedRBM(nn.Module):
    def __init__(self, n_nodes=[784, 256, 128], k=1):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_visible = n_nodes[0]
        rbm_modules = []
        self.k = k

        for i in range(len(self.n_nodes) - 1):
            n_visible = n_nodes[i]
            n_hidden = n_nodes[i+1]
            rbm = RBM(n_visible, n_hidden, self.k)
            rbm_modules.append(rbm)
        self.rbm_modules = nn.ModuleList(rbm_modules)

    def _pass(self, v):
        h_prob = v
        for _, model in enumerate(self.rbm_modules):
            h_prob, h_sample = model._pass(h_prob)
        return h_prob, h_sample

    def _reverse_pass(self, h):
        v_prob = h
        for _, model in reversed(list(enumerate(self.rbm_modules))):
            v_prob, v_sample = model._reverse_pass(v_prob)
        return v_prob, v_sample

    def forward(self, input):
        h_prob, _ = self._pass(input)
        for _ in range(self.k):
            v_prob, _ = self._reverse_pass(h_prob)
            h_prob, _ = self._pass(v_prob)
        return v_prob, h_prob
    
    def fit(self, input, lr, batch_size):
        loss = 0
        v = input
        for _, model in enumerate(self.rbm_modules):
            loss += model.constrastive_divergence(v,
                                                  lr=lr, 
                                                  batch_size=batch_size)
            _, v = model(v)
        return loss
