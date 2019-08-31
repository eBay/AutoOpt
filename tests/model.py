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

import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(10, 1)

    def forward(self, input):
        self.fc.A_prev = input.t()
        return nn.functional.softmax(self.fc(input), dim=1)
