"""Embedding layer."""

from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn

class Embedding(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding_table = nn.Parameter(
            data=torch.clamp(
                input=torch.randn(vocab_size, embedding_dim),
                min=-3,
                max=3
            )
        )
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table[token_ids]
