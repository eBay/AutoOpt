import torch.nn as nn


class TestModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(10, 1)

    def forward(self, input):
        self.fc.A_prev = input.t()
        return nn.functional.softmax(self.fc(input), dim=1)
