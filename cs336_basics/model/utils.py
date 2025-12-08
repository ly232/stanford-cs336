"""Custom implementations of various helpers"""

from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn import functional as F

import math
import torch
import torch.nn as nn

def softmax(x: Tensor, dim: int) -> Tensor:
    """Compute softmax over the specified dimension.
    
    Returns a tensor of same shape.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    result = exp_x / exp_x.sum(dim=dim, keepdim=True)
    assert x.shape == result.shape
    return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],  # (..., n, dk)
    K: Float[Tensor, " ... keys d_k"],  # (..., m, dk)
    V: Float[Tensor, " ... values d_v"],  # (..., m, dv)
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Note:
    - m & n are the same as seq_len for self-attention (decoder-only arch).
    - m == keys == values, n == queries for cross-attention (enc+dec arch).

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    softmax_arg = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    softmax_arg /= math.sqrt(d_k)
    if mask is not None:
        softmax_arg = softmax_arg.masked_fill(~mask, float('-inf'))
    attn = softmax(softmax_arg, dim=-1)  # (..., n, m)
    return einsum(attn, V, "... n m, ... m d_v -> ... n d_v")
