import torch
from torch import nn
from enum import Enum
from typing import Type
from models.activation import Activation

class ResBlock(nn.Module):
    def __init__(self, n_hidden, activation:Activation):
        super().__init__()
        self.fc = nn.Linear(n_hidden, n_hidden)
        self.activation = Activation.func(activation)()

    def forward(self, input):
        h = input + self.fc(input)
        h = self.activation(h)
        return h