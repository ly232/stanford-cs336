"""SwiGLU for positionwise feedforward."""

from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn

class PositionwiseFeedforward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(data=torch.zeros(d_ff, d_model))
        self.w2 = nn.Parameter(data=torch.zeros(d_model, d_ff))
        self.w3 = nn.Parameter(data=torch.zeros(d_ff, d_model))

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.multiply(x, torch.sigmoid(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            self.w2,
            torch.multiply(
                self._silu(einsum(self.w1, x, "ff model, batch seq model -> batch seq ff")),
                einsum(self.w3, x, "ff model, batch seq model -> batch seq ff")
            ),
            "model ff, batch seq ff -> batch seq model"
        )