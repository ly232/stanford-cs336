"""SGD Optimizer

This is copied from assignment specs. Adding here for posterity, and to capture
learning notes.

ER diagram:

                    +----------------------+
                    |   torch.optim        |
                    |     Optimizer        |
                    +----------------------+
                               |
                               | subclass
                               v
                    +----------------------+
                    |        SGD           |
                    |----------------------|
                    | - param_groups       |
                    | - state              |
                    +----------------------+
                         |           |
            1 : n        |           |   1 : n
                         v           v
              +----------------+   +----------------+
              |  ParamGroup    |   |     State      |
              |----------------|   |----------------|
              | - lr           |   | - param        |
              | - params       |   | - timestep     |
              +----------------+   +----------------+
                     |
         1 : n       |
                     v
              +----------------+
              |     Param      |
              |----------------|
              | - data         |
              | - grad         |
              +----------------+


[torch.optim.Optimizer]<:-[SGD|param_groups|state]
[SGD]->1:n[ParamGroup|lr|params]
[ParamGroup]1:n->[Param|data|grad]
[SGD]->[State|param|timestep]
[State]->[Param]
"""

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import pytest

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']  # get learning rate

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # get state associated with p.
                t = state.get('t', 0)  # get iter number from state, or init val
                grad = p.grad.data  # get grad of loss wrt p.
                p.data -= lr / math.sqrt(t + 1) * grad  # update weight tensor
                state['t'] = t + 1  # increment iter
        
        return loss

@pytest.mark.parametrize("lr", [1e1, 1e2, 1e3])
def test_sgd_lr1e1(lr):
    weights  = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)

    for t in range(10):
        opt.zero_grad()  # reset grads
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()  # backprop to compute grad
        opt.step()  # run opt step
