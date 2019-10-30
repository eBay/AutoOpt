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
from autoopt import AutoOptError


class AutoGaussNewton(AutoOptimizer):

    def __init__(self, model, beta2=0.0, eps=1.0, weight_decay=0.0, amsgrad=False, ewma=0.9, gamma0=0.999):
        if not 0.0 <= beta2 < 1.0:
            raise AutoOptError('Error: Beta[1] should have a value in range [0.0, 1).')
        if eps < 0.0:
            raise AutoOptError('Error: Epsilon cannot have a negative value.')

        self.model = model
        defaults = {
                    'beta2': beta2,
                    'weight_decay': weight_decay,
                    'eps': eps,
                    'amsgrad': amsgrad,
                    'ewma': ewma,
                    'gamma0': gamma0
                    }
        super(AutoGaussNewton, self).__init__(model, defaults)

    def __setstate__(self, state):
        super(AutoGaussNewton, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            verbose: Be verbose
        """
        super(AutoGaussNewton, self).step(closure=closure)

        loss = None
        if closure is not None:
            loss = closure()

        self.model.auto_params = {'lr': [], 'momentum': []}
        max_exp_avg_sq = 0

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                if torch.sum(torch.abs(param.grad)) == 0:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta2 = group['beta2']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], param.data)

                # Decay the first and second moment running average coefficient
                exp_avg_sq_all = torch.zeros_like(param.grad_all)
                exp_avg_sq_all.mul_(beta2).addcmul_(1 - beta2, param.grad_all, param.grad_all)

                if len(param.grad_all.size()) == 2:
                    exp_avg_sq_all = exp_avg_sq_all[self.f_w_x != 0] / self.f_w_x[self.f_w_x != 0].unsqueeze(1)
                elif len(param.grad_all.size()) == 3:
                    exp_avg_sq_all = exp_avg_sq_all[self.f_w_x != 0] / \
                                     self.f_w_x[self.f_w_x != 0].unsqueeze(1).unsqueeze(1)
                elif len(param.grad_all.size()) == 5:
                    exp_avg_sq_all = exp_avg_sq_all[self.f_w_x != 0] / \
                                     self.f_w_x[self.f_w_x != 0].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                else:
                    raise AutoOptError('Error: Unhandled grad_all size.')

                exp_avg_sq = exp_avg_sq_all.sum(0) / (2 * self.f_w_x.shape[0])

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.add_(group['eps'])
                else:
                    denom = exp_avg_sq.add_(group['eps'])

                self.auto_tune(parameter=param, hessian=denom, verbose=verbose)

                group['lr'] = 1 - param.gamma[0]
                beta1 = param.gamma[1] / (1 - param.gamma[0])

                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                param.data.addcdiv_(-step_size, exp_avg, denom)

                self.model.auto_params['lr'].append(group['lr'].item())
                self.model.auto_params['momentum'].append(beta1.item())

        return loss
