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
import autoopt
from examples.mnist import get_args, get_data, get_optimizer, get_model, CNN, FCNet


class TestMNIST(unittest.TestCase):

    def test_get_args(self):
        args = get_args(['--cuda', '--model', 'cnn', '--optimizer', 'auto-sgd'])
        self.assertEqual(args.cuda, torch.cuda.is_available())
        self.assertEqual(args.model, 'cnn')
        self.assertEqual(args.optimizer, 'auto-sgd')

    def test_get_data(self):
        args = get_args(['--data-folder', 'examples/data'])
        train_loader, test_loader = get_data(args)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        self.assertEqual(len(train_loader.dataset), 60000)
        self.assertEqual(len(test_loader.dataset), 10000)

    def test_get_model(self):
        args = get_args(['--model', 'cnn'])
        model = get_model(args)
        self.assertIsInstance(model, CNN)

        args = get_args(['--model', 'fc'])
        model = get_model(args)
        self.assertIsInstance(model, FCNet)

    def test_get_optimizer(self):
        args = get_args(['--model', 'cnn'])
        model = get_model(args)
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, torch.optim.SGD)

        args = get_args(['--optimizer', 'adam'])
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, torch.optim.Adam)

        args = get_args(['--optimizer', 'adagrad'])
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, torch.optim.Adagrad)

        args = get_args(['--optimizer', 'auto-sgd'])
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, autoopt.optim.AutoSGD)

        args = get_args(['--optimizer', 'auto-adam'])
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, autoopt.optim.AutoAdam)

        args = get_args(['--optimizer', 'auto-adagrad'])
        optimizer = get_optimizer(args, model)
        self.assertIsInstance(optimizer, autoopt.optim.AutoAdagrad)


if __name__ == '__main__':
    unittest.main()
