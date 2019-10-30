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


class AutoAdagrad(AutoOptimizer):
    """Implements AutoAdagrad algorithm.

    Arguments:
        model (torch.nn.Module): Model containing the parameters to optimize
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, model, weight_decay=0, initial_accumulator_value=0, ewma=0.9, gamma0=0.999):
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value,
                        ewma=ewma, gamma0=gamma0)
        super(AutoAdagrad, self).__init__(model, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p.data, initial_accumulator_value)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            verbose: Print debug messages if set to True.
        """
        super(AutoAdagrad, self).step(closure=closure)

        loss = None
        if closure is not None:
            loss = closure()

        self.model.auto_params = {'lr': [], 'momentum': []}

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if param.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(group['weight_decay'], param.data)

                if grad.is_sparse:
                    raise NotImplementedError
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    self.auto_tune(parameter=param, hessian=std, with_momentum=True, verbose=verbose)

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
                            buf.mul_(momentum).add_(1 - momentum, grad)  # should be modified
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, grad)

                            grad = buf

                    param.data.addcdiv_(-group['lr'], grad, std)

        return loss
