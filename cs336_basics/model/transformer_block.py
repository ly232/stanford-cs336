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

        self.d_k, self.d_v = d_model // num_heads, d_model // num_heads

        self.mha_prenorm = RmsLayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(
            d_model, num_heads, seq_len, theta)
        
        self.pff_prenorm = RmsLayerNorm(d_model)
        self.pff = PositionwiseFeedforward(d_model, d_ff)

    def load_weights(self, q_proj, k_proj, v_proj, output_proj, 
                     ln1, ffn_w1, ffn_w2, ffn_w3, ln2):
        d_k, d_v = self.d_k, self.d_v
        # Load MHA parameters:
        for i, head in enumerate(self.mha.heads):
            head.queries.load_state_dict(
                {'weights': q_proj[i*d_k:(i+1)*d_k, :]}, strict=False)
            head.keys.load_state_dict(
                {'weights': k_proj[i*d_k:(i+1)*d_k, :]}, strict=False)
            head.values.load_state_dict(
                {'weights': v_proj[i*d_v:(i+1)*d_v, :]}, strict=False)
        self.mha.projection.load_state_dict(
            {'weights': output_proj}, strict=False)
        # Load RMS norm layers parameters:
        self.mha_prenorm.load_state_dict({'weights': ln1}, strict=False)
        self.pff_prenorm.load_state_dict({'weights': ln2}, strict=False)
        # Load positionwise feedforward network's parameters:
        self.pff.load_state_dict({
            'w1': ffn_w1,
            'w2': ffn_w2,
            'w3': ffn_w3,
        }, strict=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        """
        In: batch sequence_length d_model
        Out: batch sequence_length d_model
        """
        y = x + self.mha(self.mha_prenorm(x), token_positions)  # attention is linear (in V).
        y = y + self.pff(self.pff_prenorm(y))  # to add non-linearity.
        return y
