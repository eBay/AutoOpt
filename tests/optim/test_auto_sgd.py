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

import unittest
import torch
import torch.nn.functional as torch_fx
from autoopt.optim import AutoSGD
from tests.model import TestModel


class TestAutoSGD(unittest.TestCase):

    def test_auto_sgd(self):
        model = TestModel()
        model.train()
        optimizer = AutoSGD(model=model)
        optimizer.zero_grad()
        output = model(torch.rand(8, 10))
        model.loss_all = torch_fx.nll_loss(output, torch.tensor([0] * 8), reduction='none')
        loss = torch.mean(model.loss_all)
        loss.backward()
        optimizer.step()
