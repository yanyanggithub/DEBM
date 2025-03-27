import torch
from torch import nn
from torch.nn import functional as F

def initialize_weights(m):
    """
    Initializes the weights of an RBM using Glorot/Xavier initialization.
    """
    if isinstance(m, nn.Linear):  # Apply to linear layers (weight matrix)
        nn.init.xavier_uniform_(m.weight)
        # You can also initialize biases if needed:
        nn.init.zeros_(m.bias)


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine
    """
    def __init__(self, n_visible, n_hidden, k, 
                 sparsity=0.01, dropout_prob=0.2, device="cpu"):
        super().__init__()

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01) # Initial value for Xavier initialization
        self.v_bias = nn.Parameter(torch.zeros(1, n_visible))
        self.h_bias = nn.Parameter(torch.zeros(1, n_hidden))
        self.k  = k
        self.sparsity = sparsity  # Regularization term for hidden units

        # Add L1 regularization to the weights (optional)
        self.l1_weight = nn.Parameter(torch.zeros(n_hidden, n_visible))
        self.dropout_prob = dropout_prob
        self.device = device

        # Apply weight initialization after creation
        self.apply(initialize_weights)  # Crucial: Applies the init function to all submodules
    
    def _sample(self, prob):
        return torch.bernoulli(prob)

    def _pass(self, v):
        """
        from visible states to hidden states
        """
        # Dropout: Randomly set a fraction of the hidden units to zero
        mask = (torch.rand(v.shape[0], 1) < self.dropout_prob).to(self.device)
        h_bias = self.h_bias * mask  # Apply dropout to bias

        h_prob = F.linear(v, self.weight, h_bias)
        h_prob = torch.sigmoid(h_prob) # or other activation function
        return h_prob, self._sample(h_prob)
    
    def _reverse_pass(self, h):
        """
        from hidden states to visible states
        """
        mask = (torch.rand(h.shape[0], 1) < self.dropout_prob).to(self.device)
        mask = mask.to(self.device)
        v_bias = self.v_bias * mask  # Apply dropout to bias

        v_prob = torch.sigmoid(F.linear(h, self.weight.t(), v_bias))
        return v_prob, self._sample(v_prob)
    
    def contrastive_divergence(self, X, lr=0.01, batch_size=64):
        """
        Performs one step of Contrastive Divergence learning.
        """
        # 1. Positive Phase:  Sample from the current model
        pos_h_prob, pos_h = self._pass(X)

        # 2. Negative Phase (k steps of CD): Sample and update
        h = pos_h
        for _ in range(self.k):
            v_recon_prob, v = self._reverse_pass(h)
            _, h = self._pass(v)  # Update hidden state

        # 3. Calculate Gradients
        pos_gradient = torch.matmul(pos_h_prob.t(), X)
        neg_gradient = torch.matmul(h.t(), v)

        gradient = pos_gradient - neg_gradient

        # 4. Regularization (L1 regularization on hidden units)
        l1_term = self.sparsity * self.l1_weight  # Sparsity encourages zero weights
        gradient += l1_term

        # 5. Update Parameters
        gradient = gradient/batch_size
        dv_bias = torch.sum(X - v, dim=0)/batch_size
        dh_bias = torch.sum(pos_h_prob - h, dim=0)/batch_size
        with torch.no_grad():
            self.weight += lr * gradient
            self.v_bias += lr * dv_bias
            self.h_bias += lr * dh_bias

        # 6. Calculate Loss (Mean Squared Error between reconstructed and original data)
        loss = torch.mean(torch.sum((X - v_recon_prob)**2, dim=1)) # Mean squared error
        return loss
    
    def forward(self, v):
        h_prob, _ = self._pass(v)
        for _ in range(self.k):
            v_prob, _ = self._reverse_pass(h_prob)
            h_prob, _ = self._pass(v_prob)
        return v_prob, h_prob

