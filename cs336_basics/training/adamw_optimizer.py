"""AdamW Optimizer."""

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, weight_decay, betas, eps):
        super().__init__(params, {
            'lr': lr,
            'weight_decay': weight_decay,
            'betas': betas,
            'eps': eps,
        })

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                t = state.get('t', 1)

                # gradient
                g = p.grad.data

                # first moment
                m = state.get('m', torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * g

                # second moment
                v = state.get('v', torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * (g ** 2)

                # adjusted alpha
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # perform weight update
                p.data -= alpha_t * \
                    torch.divide(m, torch.sqrt(v) + eps)
                
                # apply weight decay
                p.data -= alpha * weight_decay * p.data

                # update state
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
