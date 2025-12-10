"""Rotary Positional Embedding."""

from einops import einsum, rearrange
from torch.nn import functional as F

import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.device = device

        # Pre-compute angles.
        angles = torch.stack(
            [
                torch.tensor(
                    [i / (self.theta ** (2 * k / self.d_k))
                     for k in range(self.d_k // 2)])
            for i in range(max_seq_len)
        ])  # (seq_len, d_k // 2)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Processes an input tensor of shape (..., seq_len, d_k).

        Returns a tensor of same shape. token_positions has shape (..., seq_len)
        to indicate position of x's elements alone the seq_len dimension.
        """
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2])
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        # broadcasting trick:
        # - self.sin is (seq, d//2)
        # - token_positions is (..., seq)
        # - broadcast turns self.sin[token_positions] to (..., seq, d//2)
        sin, cos = self.sin[token_positions], self.cos[token_positions]
        x_rot_even = torch.multiply(x_even, cos) - torch.multiply(x_odd, sin)
        x_rot_odd = torch.multiply(x_even, sin) + torch.multiply(x_odd, cos)
        return rearrange(
            # stack even and odd tensors and put the new dimension at end. this
            # produces a new tensor of shape (..., seq, dk//2, 2)
            torch.stack([x_rot_even, x_rot_odd], dim=-1),
            # interleaves by "concat" each pair in the last 2 dimensions.
            "... seq half_d two -> ... seq (half_d two)"
        )
        
