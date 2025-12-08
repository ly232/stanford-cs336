"""Transformer block."""

from einops import einsum
from torch.nn import functional as F

from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.model.positionwise_feedforward import PositionwiseFeedforward
from cs336_basics.model.rms_layer_norm import RmsLayerNorm

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ff, seq_len, theta=None):
        super().__init__()

        self.mha_prenorm = RmsLayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, seq_len, theta)
        
        self.pff_prenorm = RmsLayerNorm(d_model)
        self.pff = PositionwiseFeedforward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.mha(self.mha_prenorm(x))  # attention is linear (in V).
        y = y + self.pff(self.pff_prenorm(y))  # to add non-linearity.
        return y
