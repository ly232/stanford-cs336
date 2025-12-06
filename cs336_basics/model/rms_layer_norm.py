"""RMS Layer Norm."""

from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn

class RmsLayerNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(data=torch.ones(d_model))

    def _rms_norm(self, a: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            1 / self.d_model * (torch.sum(torch.square(a), dim=-1) + self.eps))
        # Note: a has shape (batch_sz, seq_len, d_model), rms has shape (batch_sz, seq_len)
        # torch.unsqueeze(2) adds a dimension at position 2 to make rms (batch_sz, seq_len, 1),
        # then we can let broadcast do the magic.
        return torch.multiply(torch.divide(a, rms.unsqueeze(2)), self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = self._rms_norm(x)
        return result.to(in_dtype)
