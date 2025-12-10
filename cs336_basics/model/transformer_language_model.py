"""Transformer language model."""

from einops import einsum
from torch.nn import functional as F

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rms_layer_norm import RmsLayerNorm
from cs336_basics.model.transformer_block import TransformerBlock
from cs336_basics.model.utils import softmax

import torch
import torch.nn as nn

class TransformerLanguageModel(nn.Module):
    
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float):
        super().__init__()
        
        self.token_embedding = Embedding(vocab_size=vocab_size, embedding_dim=d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, 
                             context_length, rope_theta)
            for _ in range(num_layers)
        ])

        self.final_mlp_prenorm = RmsLayerNorm(d_model)

        self.final_mlp = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions= torch.arange(x.shape[-1])
        out = self.token_embedding(x)
        for tb in self.transformer_blocks:
            out = tb(out, token_positions)
        out = self.final_mlp_prenorm(out)
        out = self.final_mlp(out)
        # return softmax(out, dim=-1)  # <-- tests want unnormalized output.
        return out
