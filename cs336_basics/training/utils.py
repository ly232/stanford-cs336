from jaxtyping import Bool, Float, Int
from torch import Tensor

import torch

def cross_entropy(
        inputs: Float[Tensor, 'batch_size vocab_size'],
        targets: Int[Tensor, 'batch_size'])\
    -> Float[Tensor, '']:
    """Computes cross-entropy loss of given tensor, averaged over batch.

    LSE (log-sum-exp) trick:
    - log(sum_i(e^i)) = log(e^m*sum_i(e^(i-m)))) = log(e^m)+log(sum_i(e^(i-m))).
    - then we can let m = max(i), to help with numeric stability.
    """
    max_inputs = inputs.max(dim=-1, keepdim=True).values
    lse = max_inputs + torch.log(
        torch.sum(
            torch.exp(inputs - max_inputs), dim=-1, keepdim=True))
    # gather: for every output position, pick exactly one element along dim.
    target_logits = inputs.gather(dim=1, index=targets.unsqueeze(1))
    loss =  lse - target_logits
    return loss.mean(dim=0)
