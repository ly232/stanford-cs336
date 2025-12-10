"""Multi-head self attention."""

from cs336_basics.model.linear import Linear
from cs336_basics.model.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.model.utils import scaled_dot_product_attention
from einops import einsum
from torch.nn import functional as F

import torch
import torch.nn as nn


class SingleHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_v: int, seq_len: int, theta: float | None = None, token_positions: torch.Tensor | None = None):
        super().__init__()
        self.rotary_positional_embedding = None
        if theta is not None:
            self.rotary_positional_embedding = RotaryPositionalEmbedding(
                theta, d_k, seq_len)
        self.queries = Linear(in_features=d_model, out_features=d_k)
        self.keys = Linear(in_features=d_model, out_features=d_k)
        self.values = Linear(in_features=d_model, out_features=d_v)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        """
        In shape: (..., seq_len, d_model)
        Out shape: (..., seq_len, d_v)
        """
        queries = self.queries(x)  # (..., seq_len, d_k)
        keys = self.keys(x)  # (..., seq_len, d_k)
        values = self.values(x)  # (..., seq_len, d_v)
        if self.rotary_positional_embedding is not None:
            queries = self.rotary_positional_embedding(queries, token_positions)
            keys = self.rotary_positional_embedding(keys, token_positions)
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        return scaled_dot_product_attention(
            queries, keys, values, causal_mask)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, seq_len: int, theta: float | None=None, token_positions: torch.Tensor | None = None):
        super().__init__()

        # Attention layers
        d_k, d_v = d_model // num_heads, d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(
                d_model, d_k, d_v, seq_len, theta, token_positions)
            for _ in range(num_heads)
        ])

        # Final projection (`Wo` in original spec) layers.
        #
        # Note that projection layer is not the same as the FFN MLP layer. That
        # is not part of the multihead attention block. RMS norm layer is also
        # not part of multihead attention.
        self.projection = Linear(
            in_features=num_heads*d_v, out_features=d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        """
        In: (..., seq_len, d_model)
        Out: (..., seq_len, d_model)
        """
        # Each head produces h(x) which is (..., seq_len, dv).
        # Concat together all num_heads of them along the -1 dimension,
        # so that dimension -1 has total length dv * num_heads = d_model.
        out = torch.cat([h(x, token_positions) for h in self.heads], dim=-1)  # (..., seq_len, d_model)
        # Apply projection.
        out = self.projection(out)
        return out
