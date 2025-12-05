"""Linear layer."""

from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn

class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = 2 / (in_features + out_features)
        self.weights = nn.Parameter(
            data=torch.clamp(
                input=torch.randn(out_features, in_features) * torch.sqrt(torch.tensor(sigma)),
                min=-3*sigma,
                max=3*sigma))
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, 
            self.weights, 
            'batch sequence d_in, d_out d_in -> batch sequence d_out')
