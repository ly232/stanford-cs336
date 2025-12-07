"""Custom implementations of various helpers"""

from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax over the specified dimension."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

