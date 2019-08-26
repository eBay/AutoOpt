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
import functools
from autoopt.optim.auto_optimizer import AutoOptimizer, store_gradients
from tests.model import TestModel


class TestAutoOptimizer(unittest.TestCase):

    def test_auto_optimizer(self):
        model = TestModel()
        defaults = {}
        auto_optimizer = AutoOptimizer(model=model, defaults=defaults)

    def test_store_gradients(self):
        model = TestModel()
        for name, layer in model._modules.items():
            if hasattr(layer, 'weight'):
                layer.register_backward_hook(functools.partial(store_gradients, name))


if __name__ == '__main__':
    unittest.main()
