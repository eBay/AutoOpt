""" 
Copyright 2019 eBay Inc.
Developers/Architects: Selcuk Kopru, Tomer Lancewicki
 
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from autoopt.optim import AutoOptimizer


class AutoSGD(AutoOptimizer):
    """Implements Automated Stochastic Gradient Descent (AutoSGD).
       During class instance creation, as compared to SGD initialization, instead of model parameters, entire model
       need to be passed to the constructor. Learning rate and momentum values are not required.

    Args:
        model (torch.nn.Module): Model containing the parameters to optimize
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = autoopt.optim.AutoSGD(model)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """
    def __init__(self, model, dampening=0, weight_decay=0, nesterov=False, ewma=0.9, gamma0=0.999):
        defaults = {
            'lr': 0.0,
            'momentum': 0.9,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
            'ewma': ewma,
            'gamma0': gamma0
        }
        super(AutoSGD, self).__init__(model, defaults=defaults)

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            verbose: Be verbose if set to True.
        """
        super(AutoSGD, self).step(closure=closure)

        loss = None
        if closure is not None:
            loss = closure()

        self.model.auto_params = {'lr': [], 'momentum': []}

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data
                d_p_all = param.grad_all.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, param.data)
                    d_p_all.add_(weight_decay, param.data)

                if torch.sum(torch.abs(d_p)) == 0:
                    continue

                self.auto_tune(parameter=param, verbose=verbose)

                if param.var_est == 0:
                    continue

                group['lr'] = 1 - param.gamma[0]
                momentum = param.gamma[1] / (1 - param.gamma[0])
                group['momentum'] = momentum
                dampening = momentum

                self.model.auto_params['lr'].append(group['lr'].item())
                self.model.auto_params['momentum'].append(momentum.item())

                if momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(1-momentum,d_p)  # should be modified
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                param.data.add_(-group['lr'], d_p)

        return loss
