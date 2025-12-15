from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import Iterable

import math
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

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # warmup
        lr = max_learning_rate * (it / warmup_iters)
    elif warmup_iters <= it <= cosine_cycle_iters:
        # cosine annealing
        lr = min_learning_rate \
            + 0.5 * (1 + math.cos(
                math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) \
            * (max_learning_rate - min_learning_rate)
    else:
        # post annealing
        lr = min_learning_rate
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grad = torch.tensor(
        [p.grad.norm(2) for p in parameters if p.grad is not None])
    l2_norm = grad.norm(2)
    if l2_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / l2_norm
